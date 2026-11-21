from submission_config import configure_runtime, validate_inputs
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from os.path import join
import numpy as np
import random
from torch_geometric.nn import GCN
import argparse
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, average_precision_score
import torchvision.transforms as tfs
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Literal
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from skimage.morphology import remove_small_objects
from monai.inferers import sliding_window_inference
from metrics import dice_torch
from dataset import Dataset2d
from custom_loss import WeightedDiceLoss, BinaryFocalLoss, masked_bce_with_logits_dice, DiceLoss1d, BCELossPosWeight, graph_smoothness_loss, SoftClDice, SoftClDiceWeighted, SoftcbDiceLoss, SymmetricGCE, ELRLossOneClass
from trainer import Trainer2d, WrappedModelBase, calc_metrics_torch
from utils_preprocess import threshold_torch, StrongAugmentation
from utils import safe_mean_torch, define_fd, dict_element_lst_append, dict_element_add, dict_element_divide, dic_to_string
from baseline_plpb import get_mask_from_uncertainty_2d
from baseline_tt_sfuda import cross_entropy_dice_2d, sigmoid_entropy_loss, entropy_loss_no_sigmoid, build_pseduo_augmentation_2d, uncert_voting_2d, uncert_voting_2d_use_pseudo, build_pseduo_augmentation_plpb_2d, flip_back_plpb_2d, build_pseudo_augmentation_flip_2d, flip_back_2d_flip, build_pseudo_augmentation_flip_with_gray_2d, build_pseudo_augmentation_flip_with_gray_2d_v2, flip_back_2d_flip_with_gray_v2
from utils import write_nii, load_yaml, write_yaml, dict_element_divide, PredictionStorage
from utils_preprocess import flip_3D_torch, rotate_2D_torch, rotate_and_flip_2D_torch, RandomContrastStretch, RandomGaussianBlur
from utils_torch_versions_of_funcs import distance_transform_edt_torch, component_weighted_distance_transform_torch
from seg_networks.graph_unet import TwoScaleGATInterp, back_gridding
from seg_networks.whitening import AugRandomContrastiveLossMoco2d, feature_map_consistency_loss
from utils import EMA_update_teacher_model, TxtLogger, set_all_seeds
from augmentations import Bezier, RandAug
from baseline_AIF_SFDA_3d.unet2d import Unet, UnetDecoder
from preprocess_build_graph_from_label_2d_goodfeature import connect_skeleton_endpoints, downsample_and_dilate
from visualize_graph import visualize_segmentation_with_points

@contextmanager
def temporary_eval(model, enabled=True):
    was_training = model.training
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(was_training)

def sample_negative_edges(edge_index, num_nodes, num_neg_samples):
    neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples, method='sparse')
    return neg_edge_index

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], dict):
            sub_dict = {}
            for sub_key in batch[0][key].keys():
                if isinstance(batch[0][key][sub_key], str):
                    sub_dict[sub_key] = [item[key][sub_key] for item in batch]
                else:
                    sub_dict[sub_key] = torch.stack([torch.tensor(item[key][sub_key]) for item in batch])
            collated_batch[key] = sub_dict
        elif isinstance(batch[0][key], str):
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = torch.stack([torch.tensor(item[key]) for item in batch])
    return collated_batch

def get_indices_of_pairs(radius, size):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))
    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))
    radius_floor = radius - 1
    full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))
    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor
    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])
    indices_to_list = []
    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])
        indices_to_list.append(indices_to)
    concat_indices_to = np.concatenate(indices_to_list, axis=0)
    return (indices_from, concat_indices_to)

def calc_agreement_from_memory_bank(prediction_storage, wanted_example):
    lst = prediction_storage[wanted_example]
    lst = list(lst)
    preds = torch.stack(lst)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    pred_1 = preds == 1
    pred_0 = preds == 0
    ent_1 = torch.sum(pred_1, dim=0).float() / preds.size(0) * torch.log2(preds.size(0) / (torch.sum(pred_1, dim=0).float() + 1e-10))
    ent_0 = torch.sum(pred_0, dim=0).float() / preds.size(0) * torch.log2(preds.size(0) / (torch.sum(pred_0, dim=0).float() + 1e-10))
    ent = ent_1 + ent_0
    return ent

def get_local_maxima_grid(points, dist_transform, d, H, W):
    """Search a square or cubic neighborhood around each node for the maximum distance-transform value."""
    dist_transform = dist_transform.T
    N = points.shape[0]
    side_length = 2 * d + 1
    grid_size = side_length * side_length
    y_coords, x_coords = torch.meshgrid(torch.arange(-d, d + 1, device=points.device, dtype=torch.float32), torch.arange(-d, d + 1, device=points.device, dtype=torch.float32), indexing='ij')
    distances = torch.sqrt(x_coords ** 2 + y_coords ** 2)
    mask = distances <= d
    x_grid = x_coords.reshape(-1)
    y_grid = y_coords.reshape(-1)
    mask_flat = mask.reshape(-1)
    x_all = points[:, 0:1] + x_grid.unsqueeze(0)
    y_all = points[:, 1:2] + y_grid.unsqueeze(0)
    x_norm = 2.0 * x_all / (W - 1) - 1.0
    y_norm = 2.0 * y_all / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    dist_transform_batch = dist_transform.unsqueeze(0).unsqueeze(0)
    grid_batch = grid.unsqueeze(1).to(dtype=torch.float32)
    sampled_values = F.grid_sample(dist_transform_batch.expand(N, 1, H, W), grid_batch, mode='nearest', padding_mode='border', align_corners=True)
    sampled_values = sampled_values.squeeze(2).squeeze(1)
    mask_expanded = mask_flat.unsqueeze(0).expand(N, -1)
    sampled_values[~mask_expanded] = -float('inf')
    local_maxima = torch.max(sampled_values, dim=1)[0]
    return local_maxima

class RefineFeature(nn.Module):

    def __init__(self, in_channels, out_channels, feature_size, radius=4, conv_layers='1'):
        super(RefineFeature, self).__init__()
        self.conv = self._make_conv_layers(in_channels, out_channels, conv_layers)
        self.relu = nn.ReLU()
        self.feature_size = feature_size
        self.ind_from, self.ind_to = get_indices_of_pairs(radius=radius, size=feature_size)
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03, betas=(0.9, 0.99))

    def _make_conv_layers(self, in_channels, out_channels, conv_layers):
        assert all((c in ['1', '3'] for c in conv_layers))
        layers = []
        for i in range(len(conv_layers)):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=int(conv_layers[i]), padding=1 if int(conv_layers[i]) == 3 else 0))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=int(conv_layers[i]), padding=1 if int(conv_layers[i]) == 3 else 0))
            if i < len(conv_layers) - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, pseudo, mask, forward_only=False):
        x = self.conv(x)
        x = self.relu(x)
        refined_feature = torch.clone(x)
        if forward_only:
            return refined_feature
        pseudo = F.interpolate(pseudo, size=self.feature_size, mode='nearest')
        mask = F.interpolate(mask, size=self.feature_size, mode='nearest')
        if x.size(2) == self.feature_size[0] and x.size(3) == self.feature_size[1]:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            raise ValueError('Feature size not match. Now: {}, Expected: {}'.format(x.size(), self.feature_size))
        x = x.view(x.size(0), x.size(1), -1)
        pseudo = pseudo.view(pseudo.size(0), pseudo.size(1), -1)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        ff = torch.index_select(x, dim=2, index=ind_from.to(device=x.device))
        ft = torch.index_select(x, dim=2, index=ind_to.to(device=x.device))
        ff_pseudo = torch.index_select(pseudo, dim=2, index=ind_from.to(device=x.device))
        ft_pseudo = torch.index_select(pseudo, dim=2, index=ind_to.to(device=x.device))
        ff_mask = torch.index_select(mask, dim=2, index=ind_from.to(device=x.device))
        ft_mask = torch.index_select(mask, dim=2, index=ind_to.to(device=x.device))
        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        ff_pseudo = torch.unsqueeze(ff_pseudo, dim=2)
        ft_pseudo = ft_pseudo.view(ft_pseudo.size(0), ft_pseudo.size(1), -1, ff_pseudo.size(3))
        ff_mask = torch.unsqueeze(ff_mask, dim=2)
        ft_mask = ft_mask.view(ft_mask.size(0), ft_mask.size(1), -1, ff_mask.size(3))
        if not forward_only:
            label = self._pseudo_mask_to_label(ff_pseudo, ft_pseudo, ff_mask, ft_mask)
            aff_cup = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1, keepdim=True))
            return (refined_feature, aff_cup, label)

    def train_refine(self, x, pseudo, mask):
        x, aff_cup, label_cup = self.forward(x, pseudo, mask)
        fg_label = label_cup[1]
        bg_label = label_cup[2]
        neg_label = label_cup[3]
        bg_count = torch.sum(bg_label) + 1e-05
        fg_count = torch.sum(fg_label) + 1e-05
        neg_count = torch.sum(neg_label) + 1e-05
        bg_loss = torch.sum(-bg_label * torch.log(aff_cup + 1e-05)) / bg_count
        fg_loss = torch.sum(-fg_label * torch.log(aff_cup + 1e-05)) / fg_count
        neg_loss = torch.sum(-neg_label * torch.log(1.0 + 1e-05 - aff_cup)) / neg_count
        loss_cup = bg_loss / 4 + fg_loss / 4 + neg_loss / 2
        return loss_cup

    def _pseudo_mask_to_label(self, ff_pseudo, ft_pseudo, ff_mask, ft_mask):
        label = torch.zeros_like(ft_mask)
        mask_valid = ff_mask * ft_mask
        mask_valid = mask_valid.float()
        label[(ff_pseudo == ft_pseudo) & (ff_pseudo == 1)] = 1
        label[(ff_pseudo == ft_pseudo) & (ff_pseudo == 0)] = 2
        label[ff_pseudo != ft_pseudo] = 3
        label = label * mask_valid
        label = F.one_hot(label.to(dtype=torch.long), num_classes=4).permute(4, 0, 1, 2, 3)
        return label

def region_aggregation(features: torch.Tensor, region_map: torch.Tensor, max_regions: int, method: Literal['mean', 'max', 'min']='mean', background_label: int=0) -> torch.Tensor:
    """Aggregate feature values within each labeled region using mean, max or min reduction."""
    B, C, M, N = features.shape
    device = features.device
    features_flat = features.permute(0, 2, 3, 1).reshape(B, -1, C)
    region_flat = region_map.reshape(B, -1)
    result = torch.zeros(B, max_regions, C, device=device, dtype=features.dtype)
    valid_mask = (region_flat >= 1) & (region_flat <= max_regions)
    for b in range(B):
        b_mask = valid_mask[b]
        if not b_mask.any():
            continue
        b_regions = region_flat[b, b_mask]
        b_features = features_flat[b, b_mask]
        region_indices = b_regions.long() - 1
        if method == 'mean':
            try:
                temp_result = torch.zeros(max_regions, C, device=device)
                temp_result.scatter_reduce_(dim=0, index=region_indices.unsqueeze(-1).expand(-1, C), src=b_features, reduce='mean', include_self=False)
                result[b] = temp_result
            except (AttributeError, RuntimeError):
                unique_regions = torch.unique(b_regions)
                for region in unique_regions:
                    if 1 <= region <= max_regions:
                        region_features = b_features[region_indices == region - 1]
                        if len(region_features) > 0:
                            result[b, region - 1] = region_features.mean(dim=0)
        elif method == 'max':
            try:
                temp_result = torch.full((max_regions, C), float('-inf'), device=device)
                temp_result.scatter_reduce_(dim=0, index=region_indices.unsqueeze(-1).expand(-1, C), src=b_features, reduce='amax', include_self=False)
                temp_result[temp_result == float('-inf')] = 0
                result[b] = temp_result
            except (AttributeError, RuntimeError):
                unique_regions = torch.unique(b_regions)
                for region in unique_regions:
                    if 1 <= region <= max_regions:
                        region_features = b_features[region_indices == region - 1]
                        if len(region_features) > 0:
                            result[b, region - 1] = region_features.max(dim=0)[0]
        elif method == 'min':
            try:
                temp_result = torch.full((max_regions, C), float('inf'), device=device)
                temp_result.scatter_reduce_(dim=0, index=region_indices.unsqueeze(-1).expand(-1, C), src=b_features, reduce='amin', include_self=False)
                temp_result[temp_result == float('inf')] = 0
                result[b] = temp_result
            except (AttributeError, RuntimeError):
                unique_regions = torch.unique(b_regions)
                for region in unique_regions:
                    if 1 <= region <= max_regions:
                        region_features = b_features[region_indices == region - 1]
                        if len(region_features) > 0:
                            result[b, region - 1] = region_features.min(dim=0)[0]
    return result

def sample_in_feature_map_2d_torch(x, coords, coords_max=None, mode='bilinear'):
    """
    tested by:
    a = torch.arange(64).reshape(1, 1, 8, 8).float()
    b = torch.tensor([[1, 3], [2, 4], [5, 6], [4, 2]]).unsqueeze(0).float()
    sample_in_feature_map_2d_torch(a, b)

    expected: [11, 20, 46, 34]
    result: [11, 20, 46, 34]

    :param x: [B, C, H, W]
    :param coords: [B, N, 2]
    :param coords_max: int or 2-tuple, max value of coords
    :return: feature maps at the N points, [B, N, C]
    """
    _, C, H, W = x.shape
    if coords_max is None:
        coords_max = (H, W)
    elif isinstance(coords_max, int):
        coords_max = (coords_max, coords_max)
    coords = coords.to(x.device, dtype=x.dtype)
    coords_max_arr = torch.tensor(coords_max, dtype=torch.float32, device=x.device).reshape([1, 1, 2])
    coords = (coords + 0.5) / coords_max_arr * 2 - 1
    B, N, _ = coords.shape
    coords = torch.flip(coords, dims=[2])
    coords = coords.reshape([B, N, 1, -1])
    output_features = F.grid_sample(x, coords, mode=mode)
    output_features = output_features.reshape([B, C, N]).permute([0, 2, 1])
    return output_features

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.track_running_stats = False

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w x d
        output: batch_size x 1 x h x w x d
    """
    assert v.dim() == 5
    n, c, h, w, d = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * d * np.log2(c))

def _do_train_refine_feature(model, refine_feature_model, feature_map_from, x, uncertainty_mask, pseudo_label, lr=0.03, weight_decay=1e-05):
    optimizer = torch.optim.Adam(refine_feature_model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=weight_decay)
    optimizer.zero_grad()
    device = x.device
    with torch.no_grad():
        pred, leak_dic = model(x)
        feature_map = leak_dic[feature_map_from]
    pseudo_label = pseudo_label[:, 1:2]
    uncertainty_mask = uncertainty_mask[:, 1:2]
    loss = refine_feature_model.train_refine(feature_map, pseudo_label, uncertainty_mask)
    loss.backward()
    optimizer.step()
    losses_dic = {'loss_refine': loss.item()}
    return losses_dic

def pseudo_mask_from_prediction(pred, thres=0.1):
    assert 0 < thres < 0.5
    thres_hi = 1 - thres
    mask = torch.zeros_like(pred)
    mask[pred > thres_hi] = 1
    mask[pred < thres] = 1
    pseudo = (pred > 0.5).to(dtype=pred.dtype)
    return (pseudo, mask)

class WrappedModel(WrappedModelBase):

    def forward(self, x):
        final_pred, _ = self.model(x)
        final_pred = torch.sigmoid(final_pred)
        return final_pred

class WrappedModelNoSigmoid(WrappedModelBase):

    def forward(self, x):
        final_pred, _ = self.model(x)
        return final_pred

class ModuleForSegmentorTest(nn.Module):

    def __init__(self, model, refiner, gnn, segmentor, feature_map_from, refiner_feature_map_from, interp_mode='bilinear', merge_mode='add', scales='12'):
        super(ModuleForSegmentorTest, self).__init__()
        self.model = model
        self.refiner = refiner
        self.gnn = gnn
        assert gnn.__class__.__name__ in ['TwoScaleGATInterp']
        self.segmentor = segmentor
        self.feature_map_from = feature_map_from
        self.refiner_feature_map_from = refiner_feature_map_from
        self.interp_mode = interp_mode
        self.merge_mode = merge_mode
        self.scales = scales

    def forward(self, x, gnn_info):
        pred, leak_dic = self.model(x)
        feature_map = leak_dic[self.refiner_feature_map_from]
        refined_feature = self.refiner(feature_map, torch.zeros_like(pred[:, 1:2]), torch.zeros_like(pred[:, 1:2]), forward_only=True)
        gnn_feature = [sample_in_feature_map_2d_torch(refined_feature, gnn_info[0]['coords'], mode=self.interp_mode)[0], sample_in_feature_map_2d_torch(refined_feature, gnn_info[1]['coords'], mode=self.interp_mode)[0]]
        pred_gnn_1, pred_gnn_2 = self.gnn(feature1=gnn_feature[0], edge_index1=gnn_info[0]['edges'].squeeze(0).T, edge_attr1=gnn_info[0]['edge_weights'].squeeze(0), pos1=gnn_info[0]['coords'].squeeze(0), feature2=gnn_feature[1], edge_index2=gnn_info[1]['edges'].squeeze(0).T, edge_attr2=gnn_info[1]['edge_weights'].squeeze(0), pos2=gnn_info[1]['coords'].squeeze(0))
        pred_gnn_1 = torch.sigmoid(pred_gnn_1).squeeze(-1)
        pred_gnn_2 = torch.sigmoid(pred_gnn_2).squeeze(-1)
        gnn_info[0]['pred'] = pred_gnn_1
        gnn_info[1]['pred'] = pred_gnn_2
        gnn_info[0]['coords'] = gnn_info[0]['coords'][0]
        gnn_info[1]['coords'] = gnn_info[1]['coords'][0]
        feature_map = prepare_feature_map_for_segmentor_2d(x, self.feature_map_from, leak_dic, gnn_dic_list=gnn_info, device=x.device, mode=self.merge_mode, scales=self.scales)
        if self.feature_map_from == 'map_0':
            segmentor_pred = self.segmentor(feature_map, [leak_dic['skip_3'], leak_dic['skip_2'], leak_dic['skip_1'], leak_dic['skip_0']], x_map_num=0)
        elif self.feature_map_from == 'map_1':
            segmentor_pred = self.segmentor(feature_map, [leak_dic['skip_2'], leak_dic['skip_1'], leak_dic['skip_0']], x_map_num=1)
        elif self.feature_map_from == 'map_2':
            segmentor_pred = self.segmentor(feature_map, [leak_dic['skip_1'], leak_dic['skip_0']], x_map_num=2)
        elif self.feature_map_from == 'map_3':
            segmentor_pred = self.segmentor(feature_map, [leak_dic['skip_0']], x_map_num=3)
        elif self.feature_map_from == 'map_4':
            segmentor_pred = self.segmentor(feature_map, [], x_map_num=4)
        segmentor_pred = torch.sigmoid(segmentor_pred)
        return (segmentor_pred, None)

def make_net_2d(net_conf='FIVES', norm_layer='instance', do_fix_bn=True, checkpoint=None, image_size=(1024, 1024)):
    norml = {'instance': nn.InstanceNorm2d, 'batch': nn.BatchNorm2d}[norm_layer]
    state_dict_path = checkpoint or os.environ.get('VESSELGDA_SOURCE_CHECKPOINT', 'checkpoints/fives_batch.pth')
    if not osp.isfile(state_dict_path):
        raise FileNotFoundError(f'Source checkpoint missing: {state_dict_path}. Set source_checkpoint in the YAML; weights are not bundled.')
    unet = Unet(input_nc=3, output_nc=1, norm_layer=norml, last_layer='Identity', output_mode=Unet.ALL_OUTPUT, check_input_shape=tuple(image_size))
    unet.load_state_dict(torch.load(state_dict_path, map_location='cpu'), strict=True)
    if do_fix_bn:
        unet.apply(fix_bn)
    return unet


def prepare_feature_map_for_segmentor_2d(x, feature_map_from, leak_dic, gnn_dic_list, device, mode='add', scales='12'):
    feature_map_this = leak_dic[feature_map_from]
    grid_init_im_size = 128
    input_im_size = x.shape[-1]
    gnn_grid_0 = back_gridding(torch.tensor(gnn_dic_list[0]['pred'], device=device).unsqueeze(1), torch.tensor(gnn_dic_list[0]['coords'], device=device) / input_im_size * grid_init_im_size, im_size=(grid_init_im_size, grid_init_im_size))
    gnn_grid_1 = back_gridding(torch.tensor(gnn_dic_list[1]['pred'], device=device).unsqueeze(1), torch.tensor(gnn_dic_list[1]['coords'], device=device) / input_im_size * grid_init_im_size, im_size=(grid_init_im_size, grid_init_im_size))
    gnn_grid_0 = F.interpolate(gnn_grid_0.unsqueeze(0).to(device=device, dtype=torch.float32), size=feature_map_this.shape[2:], mode='bilinear', align_corners=False)
    gnn_grid_1 = F.interpolate(gnn_grid_1.unsqueeze(0).to(device=device, dtype=torch.float32), size=feature_map_this.shape[2:], mode='bilinear', align_corners=False)
    if '1' in scales:
        if mode == 'add':
            feature_map = feature_map_this + gnn_grid_0
        elif mode == 'att':
            feature_map = feature_map_this * (1 + gnn_grid_0)
    elif '2' in scales:
        if mode == 'add':
            feature_map = feature_map_this + gnn_grid_1
        elif mode == 'att':
            feature_map = feature_map_this * (1 + gnn_grid_1)
    return feature_map

class TrainerGNN(Trainer2d):

    def __init__(self, model, optimizer, criterion, device, train_loader, test_loader, **kwargs):
        super().__init__(model, optimizer, criterion, device, train_loader, test_loader, **kwargs)
        self.max_epoch = kwargs['max_epoch']
        self.test_loader = test_loader
        if kwargs.get('patch_size_description', 'large') == 'large':
            in_channels_dic = {'map_5': (32, 32), 'map_4': (64, 64), 'map_3': (128, 64), 'map_2': (256, 128), 'map_1': (320, 128), 'map_0': (320, 128), 'skip_4': (1024, 128), 'skip_3': (512, 128), 'skip_2': (256, 128), 'skip_1': (128, 64), 'skip_0': (64, 64), 'input': (3, 64)}
        elif kwargs.get('patch_size_description', 'large') == 'small':
            in_channels_dic = {'map_4': (32, 32), 'map_3': (64, 64), 'map_2': (128, 128), 'map_1': (256, 128), 'map_0': (320, 128)}
        else:
            raise ValueError('patch_size_description should be large or small')
        self.in_channels_dic = in_channels_dic
        if kwargs['gnn'].get('pred_edge', False):
            out_channels = 32
        else:
            out_channels = 1
        if kwargs['gnn'].get('type') != 'GATInterp':
            raise NotImplementedError('This release includes the two-scale GATInterp teacher used by the method.')
        assert kwargs['gnn'].get('scales', 1) == 2
        self.gnn = TwoScaleGATInterp(in_channels=in_channels_dic[kwargs['gnn']['map_from']][0], hidden_channels=kwargs['gnn'].get('hidden_ch', in_channels_dic[kwargs['gnn']['map_from']][1]), out_channels=out_channels, num_layers=kwargs['gnn'].get('num_layers', 2), v2=kwargs['gnn'].get('v2', False), heads=kwargs['gnn']['heads'], feature_down=kwargs['gnn']['feature_down'], feature_up=kwargs['gnn']['feature_up'], topk=kwargs['gnn'].get('topk', None), density_correction=kwargs['gnn'].get('density_correction', 'uniform'), learn_kernel=kwargs['gnn'].get('learn_kernel', None), twod=True, residual=kwargs['gnn'].get('residual', False), norm=kwargs['gnn'].get('norm', False))
        self.gnn = self.gnn.to(device=self.device)
        self.aux_models_dict['gnn'] = self.gnn
        self.optimizer_gnn = torch.optim.Adam(self.gnn.parameters(), lr=kwargs['gnn']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=kwargs.get('weight_decay_gnn', kwargs['weight_decay']))
        self.scheduler_gnn = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_gnn, T_max=self.max_epoch, eta_min=1e-08)
        if kwargs['discriminator']['use']:
            raise NotImplementedError('This optional baseline experiment is outside this release.')
        if kwargs.get('refiner') is not None and kwargs.get('refiner').get('use'):
            if kwargs.get('gnn') is not None and kwargs.get('gnn').get('map_from') is not None:
                assert kwargs['refiner']['feature_map_from'] == kwargs['gnn']['map_from']
            refiner_kwargs = {'map_0': {'in_channels': 1024, 'out_channels': 512, 'feature_size': (64, 64), 'radius': 3}, 'map_1': {'in_channels': 512, 'out_channels': 512, 'feature_size': (128, 128), 'radius': 4}, 'map_2': {'in_channels': 256, 'out_channels': 256, 'feature_size': (256, 256), 'radius': 5}, 'map_3': {'in_channels': 128, 'out_channels': 128, 'feature_size': (512, 512), 'radius': 5}, 'map_4': {'in_channels': 64, 'out_channels': 64, 'feature_size': (1024, 1024), 'radius': 6}}
            self.refiner = RefineFeature(**refiner_kwargs[kwargs['refiner']['feature_map_from']], conv_layers=kwargs['refiner'].get('conv_layers', '1'))
            self.refiner = self.refiner.to(device=self.device)
            self.aux_models_dict['refiner'] = self.refiner
            self.optimizer_refiner = torch.optim.Adam(self.refiner.parameters(), lr=kwargs['refiner']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=kwargs['weight_decay'])
            self.scheduler_refiner = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_refiner, T_max=self.max_epoch, eta_min=1e-08)
            if kwargs['refiner'].get('init_dict_path') is not None:
                self.refiner.load_state_dict(torch.load(kwargs['refiner']['init_dict_path'], map_location=torch.device('cpu')))
            if kwargs['refiner'].get('segmentor') and kwargs['refiner'].get('segmentor').get('use'):
                raise NotImplementedError('Segmentor in refiner will not be implemented in this version, please use the original code')
        if kwargs.get('training_base') == 'aif_sfda':
            raise NotImplementedError('This optional baseline experiment is outside this release.')
        if self.kwargs.get('segmentor') and self.kwargs['segmentor'].get('use'):
            self.initialize_segmentor()
        if self.kwargs['gnn'].get('gnn_memory_bank') and self.kwargs['gnn']['gnn_memory_bank'].get('use') or (self.kwargs['gnn'].get('elr') and self.kwargs['gnn']['elr'].get('use')):
            self.gnn_memory_bank = PredictionStorage(max_epoch=self.kwargs['gnn']['gnn_memory_bank']['max_memory_epoch'])
            if self.kwargs['gnn']['scales'] == 2:
                self.gnn_memory_bank_2 = PredictionStorage(max_epoch=self.kwargs['gnn']['gnn_memory_bank']['max_memory_epoch'])

    def load_models(self, model_path):
        for k, model in self.models_dict.items():
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.logger.write(f'Loaded {model_path} to model {k}')

    def load_gnn(self, gnn_path, refiner_path):
        self.gnn.load_state_dict(torch.load(gnn_path, map_location=torch.device('cpu')))
        self.logger.write(f'Loaded {gnn_path} to GNN')
        if refiner_path is not None and self.refiner is not None:
            self.refiner.load_state_dict(torch.load(refiner_path, map_location=torch.device('cpu')))
            self.logger.write(f'Loaded {refiner_path} to Refiner')

    def initialize_segmentor(self):
        self.segmentor = UnetDecoder(input_nc=3, output_nc=1, norm_layer=nn.BatchNorm2d, last_layer='Identity')
        self.segmentor = self.segmentor.to(device=self.device)
        self.segmentor.up_conv_list.load_state_dict(self.model.up_conv_list.state_dict())
        self.segmentor.up_up_list.load_state_dict(self.model.up_up_list.state_dict())
        self.segmentor.out.load_state_dict(self.model.out.state_dict())
        self.optimizer_segmentor = torch.optim.Adam(self.segmentor.parameters(), lr=self.kwargs['segmentor']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=self.kwargs['weight_decay'])
        self.scheduler_segmentor = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_segmentor, T_max=self.max_epoch, eta_min=1e-08)
        self.logger.write('Segmentor initialized.')
        self.module_for_segmentor_test = ModuleForSegmentorTest(model=self.model, refiner=self.refiner, gnn=self.gnn, segmentor=self.segmentor, feature_map_from=self.kwargs['segmentor']['feature_map_from'], refiner_feature_map_from=self.kwargs['refiner']['feature_map_from'], interp_mode=self.kwargs['segmentor'].get('interp_mode', 'bilinear'), merge_mode=self.kwargs['segmentor'].get('merge_mode', 'add'), scales=self.kwargs['segmentor'].get('scales', '12'))

    def train_epoch(self, epoch, trained_network='main'):
        self.model.train()
        total_losses = {}
        total_dice = 0
        idx = 0
        if self.kwargs.get('ema') and self.kwargs['ema_conf'].get('begin_epoch') is not None and (epoch == self.kwargs['ema_conf']['begin_epoch']):
            self.logger.write('EMA model initialized at epoch {}'.format(epoch))
            self.orig_model.load_state_dict(self.model.state_dict())
            self.teacher_model.load_state_dict(self.model.state_dict())
        with tqdm(self.train_loader) as t:
            for batch in t:
                idx += 1
                image = batch['image'].to(device=self.device, dtype=torch.float32)
                batch['label'] = None
                distmap = batch['frangi'].to(device=self.device, dtype=torch.float32)
                name = batch['name']
                gnn_dic = batch['gnn']
                gnn2_dic = batch.get('gnn2', None)
                uncertainty_mask = batch['uncertainty_mask'].to(device=self.device, dtype=torch.float32)
                pseudo_label = batch['pseudo_label'].to(device=self.device, dtype=torch.float32)
                image = self._normalize(image)
                image_lst = []
                distmap_lst = []
                gnn_dic_lists = []
                uncertainty_mask_lst = []
                pseudo_label_lst = []
                image_lst.append(image)
                distmap_lst.append(distmap)
                gnn_dic_lists.append([gnn_dic, gnn2_dic] if gnn2_dic is not None else [gnn_dic])
                uncertainty_mask_lst.append(uncertainty_mask)
                pseudo_label_lst.append(pseudo_label)
                self.optimizer.zero_grad()
                if self.kwargs['discriminator']['use']:
                    discriminator_configs = {'net': self.discriminator, 'weight': self.kwargs['discriminator']['weight']}
                else:
                    discriminator_configs = {'net': None, 'weight': 0.5}
                if self.kwargs.get('training_base') in [None, 'plpb', 'tt_sfuda'] or trained_network == 'refiner':
                    for i in range(len(image_lst)):
                        if trained_network == 'main':
                            if self.kwargs.get('training_base') in [None, 'plpb']:
                                losses_dic = self._do_train(self.model, epoch, image_lst[i], uncertainty_mask_lst[i], pseudo_label_lst[i], self.gnn, gnn_dic_lists[i], mode='pred', lr=self.kwargs['lr'], weight_decay=self.kwargs['weight_decay'], discriminator=discriminator_configs, gnn=self.kwargs['gnn'], cpr_refine={'map': distmap_lst[i] if self.kwargs['cpr_refine']['use'] else None, 'entropy_thres': self.kwargs['cpr_refine']['thres']})
                            elif self.kwargs['training_base'] == 'tt_sfuda':
                                frangi = distmap_lst[i]
                                losses_dic = self._do_train_tt_sfuda(self.model, self.orig_model, self.gnn, image_lst[i], gnn_dic_lists[i], epoch, name, frangi=frangi)
                        elif trained_network == 'refiner':
                            losses_dic = _do_train_refine_feature(self.model, self.refiner, self.kwargs['refiner']['feature_map_from'], image_lst[i], uncertainty_mask_lst[i], pseudo_label_lst[i], lr=self.kwargs['refiner']['lr'], weight_decay=self.kwargs['weight_decay'])
                            torch.save(self.refiner.state_dict(), join(self.base_folder, f'refiner_{epoch}.pth'))
                        total_losses = dict_element_add(total_losses, losses_dic)
                t.set_postfix(dict_element_divide(total_losses, idx))
                if self.kwargs.get('ema') and (not self.kwargs['ema_conf'].get('epoch_update', True)):
                    if self.kwargs['ema_conf'].get('begin_epoch') is None or epoch >= self.kwargs['ema_conf'].get('begin_epoch', 0):
                        new_dict = EMA_update_teacher_model(self.model, self.teacher_model, self.kwargs['ema'])
                        self.teacher_model.load_state_dict(new_dict)
                        if self.kwargs.get('ema_conf', True) and self.kwargs['ema_conf'].get('update_per', 1) is not None and ((epoch + 1) % self.kwargs['ema_conf'].get('update_per', 1) == 0):
                            new_dict_orig = EMA_update_teacher_model(self.model, self.orig_model, self.kwargs['ema'])
                            self.orig_model.load_state_dict(new_dict_orig)
        if self.kwargs.get('ema') and self.kwargs['ema_conf'].get('epoch_update', True):
            if self.kwargs['ema_conf'].get('begin_epoch') is None or epoch >= self.kwargs['ema_conf'].get('begin_epoch', 0):
                new_dict = EMA_update_teacher_model(self.model, self.teacher_model, self.kwargs['ema'])
                self.teacher_model.load_state_dict(new_dict)
                if self.kwargs.get('ema_conf', True) and self.kwargs['ema_conf'].get('update_per', 1) is not None and ((epoch + 1) % self.kwargs['ema_conf'].get('update_per', 1) == 0):
                    new_dict_orig = EMA_update_teacher_model(self.model, self.orig_model, self.kwargs['ema'])
                    self.orig_model.load_state_dict(new_dict_orig)
        if self.kwargs.get('refiner') is not None and self.kwargs.get('refiner').get('use'):
            self.scheduler_refiner.step()
            self.logger.write('Refiner lr: {:.6f}'.format(self.scheduler_refiner.get_last_lr()[0]))
        self.scheduler_gnn.step()
        self.logger.write('GNN lr: {:.6f}'.format(self.scheduler_gnn.get_last_lr()[0]))
        train_str = f'Epoch {epoch}\n' + dic_to_string(dict_element_divide(total_losses, idx))
        self.logger.write(train_str)

    def _do_train(self, model, epoch, x, uncertainty_mask, pseudo_label, gnn_model, gnn_info, lr=5e-06, weight_decay=1e-05, mode='pred', refined_and_pseudo=False, reweight_pseudo=False, discriminator={'net': None, 'weight': 0.5}, gnn={'weight': 1.0, 'map_from': 'map_5', 'lr': 1e-05}, cpr_refine={'map': None, 'weight': 1.0, 'entropy_thres': 0.1, 'refine_feature_map': False}):
        assert mode in ['max', 'pred']
        if reweight_pseudo and refined_and_pseudo:
            raise ValueError('Cannot reweight pseudo label when refine_and_pseudo is True')
        self.optimizer.zero_grad()
        self.optimizer_gnn.zero_grad()
        gnn_conf = self.kwargs['gnn']
        if gnn_conf.get('delete_edge_weight_under', 0) > 0:
            for gnn_info_idx in range(len(gnn_info)):
                gnn_info[gnn_info_idx] = self._do_gnn_delete_edge_weight_under(gnn_info[gnn_info_idx], gnn_conf['delete_edge_weight_under'])
        device = x.device
        if self.kwargs.get('ema'):
            mask, pseudo_label_1 = self._calc_pseudo_mask_online(x, self.orig_model)
        else:
            mask = uncertainty_mask
            pseudo_label_1 = pseudo_label
        mask = mask.to(device=device, dtype=torch.float32)
        pseudo_label_1 = pseudo_label_1.to(device=device, dtype=torch.float32)
        x_ori = x.to(device=device, dtype=torch.float32)
        pred, leak_dic = model(x_ori)
        pred = torch.softmax(pred, dim=1)
        sh = pred.shape
        bce_loss = nn.BCELoss()
        loss_dic = {}
        ps_output = pseudo_label_1[:, 1]
        mask_output = mask[:, 1]
        loss_gnn, losses_dic = self._calc_loss_gnn(leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic)
        if self.kwargs['gnn'].get('pretrain_epochs') is not None and epoch < self.kwargs['gnn']['pretrain_epochs']:
            loss_seg = torch.tensor(0, dtype=pred.dtype, device=pred.device)
            loss_ent = torch.tensor(0, dtype=pred.dtype, device=pred.device)
            loss = loss_gnn
        else:
            if self.kwargs.get('loss_weights') is not None:
                seg_weight = self.kwargs['loss_weights']['seg']
                ent_weight = self.kwargs['loss_weights']['ent']
            else:
                Warning('Using default loss weights')
                seg_weight = 1
                if mode == 'max':
                    ent_weight = 0
                elif mode == 'pred':
                    ent_weight = 0.5
            pseudo_label = pseudo_label_1
            weight = mask
            if cpr_refine['map'] is not None:
                pseudo_label, mask = pseudo_mask_from_prediction(cpr_refine['map'], thres=cpr_refine['entropy_thres'])
            dice_loss = WeightedDiceLoss(sigmoid=False, no_bg=False)
            if mode == 'max':
                loss_seg_pixel = dice_loss(pred, pseudo_label)
            elif mode == 'pred':
                loss_seg_pixel = dice_loss(pred * weight, pseudo_label * weight)
            loss_seg = loss_seg_pixel * seg_weight
            loss_ent = entropy_loss(pred) * ent_weight
            loss = loss_seg + loss_ent + loss_gnn
            if refined_and_pseudo:
                raise ValueError('Cannot use refined_and_pseudo in this mode')
            if discriminator['net'] is not None:
                with torch.no_grad():
                    pred_discriminator = discriminator['net'](pred[:, 1:2])
                loss_discriminator = bce_loss(pred_discriminator.flatten(), torch.ones(sh[0], dtype=pred.dtype, device=pred.device))
                loss_discriminator = loss_discriminator * discriminator['weight']
                loss = loss + loss_discriminator
        loss.backward()
        if self.kwargs['gnn'].get('pretrain_epochs') is not None and epoch >= self.kwargs['gnn']['pretrain_epochs']:
            self.optimizer.step()
        self.optimizer_gnn.step()
        losses_dic.update({'loss_seg': loss_seg.item(), 'loss_ent': loss_ent.item(), 'loss_gnn': loss_gnn.item(), 'loss_discriminator': loss_discriminator.item() if discriminator['net'] is not None else 0})
        return losses_dic

    def _do_gnn_delete_edge_weight_under(self, gnn_info, thres):
        edge_weights = gnn_info['edge_weights']
        idxs = torch.where(edge_weights > thres)[1]
        new_gnn_info = {}
        for key in gnn_info.keys():
            if key in ['edges', 'edge_weights']:
                new_gnn_info[key] = gnn_info[key][:, idxs]
            else:
                new_gnn_info[key] = gnn_info[key]
        return new_gnn_info

    def _mix_frangi_and_pseudo(self, frangi, pseudo, mask, hard_frangi=False, force_pseudo_true=False):
        sigmoid_s = self.kwargs['frangi_in_pseudo']['sigmoid_s']
        sigmoid_k = self.kwargs['frangi_in_pseudo']['sigmoid_k']
        if sigmoid_k is None:
            sigmoid_k = 2.5 / sigmoid_s
        frangi = 1 / (1 + torch.exp(-(frangi - sigmoid_s) * sigmoid_k))
        if hard_frangi:
            frangi = (frangi > 0.5).to(dtype=frangi.dtype)
        if force_pseudo_true:
            mask = ((mask > 0.5) | (pseudo > 0.5)).to(dtype=mask.dtype)
        pseudo = pseudo * mask
        frangi = frangi * (1 - mask)
        return frangi + pseudo

    def _mix_frangi_and_pseudo_add_area_interest(self, frangi, pseudo, mask, hard_frangi=False, force_pseudo_true=False):
        sigmoid_s = self.kwargs['frangi_in_pseudo']['sigmoid_s']
        sigmoid_k = self.kwargs['frangi_in_pseudo']['sigmoid_k']
        frangi = frangi.unsqueeze(1)
        if sigmoid_k is None:
            sigmoid_k = 2.5 / sigmoid_s
        frangi = 1 / (1 + torch.exp(-(frangi - sigmoid_s) * sigmoid_k))
        frangi_bool = frangi > 0.5
        if hard_frangi:
            frangi = (frangi > 0.5).to(dtype=frangi.dtype)
        certain_positive_area = (pseudo > 0.5) & (mask > 0.5)
        mask[frangi_bool & ~certain_positive_area] = 0.0
        if force_pseudo_true:
            pseudo[frangi_bool & (pseudo < 0.5)] = frangi[frangi_bool & (pseudo < 0.5)]
        else:
            pseudo[frangi_bool] = frangi[frangi_bool]
        return (pseudo, mask)

    def _connect_pseudo_mask_endpoints(self, pseudo, mask, max_dist=40, angle_thresh_deg=180, connect_to_tree_thres=20):
        assert pseudo.shape[0] == 1 and pseudo.shape[1] == 1 and (len(pseudo.shape) == 4)
        pseudo_np = pseudo.detach().cpu().numpy()[0, 0]
        mask_np = mask.detach().cpu().numpy()[0, 0]
        area_interest = (pseudo_np > 0.5) | (mask_np < 0.5)
        area_interest_skeleton = skeletonize(area_interest).astype(np.float32)
        area_interest_skeleton, _ = connect_skeleton_endpoints(area_interest_skeleton, max_dist=max_dist, angle_thresh_deg=angle_thresh_deg, connect_to_tree_thres=connect_to_tree_thres)
        area_interest = downsample_and_dilate(area_interest_skeleton > 0.5, 1, 2) | (area_interest > 0.5)
        area_interest = area_interest.astype(np.float32)
        area_interest_tn_mask = (area_interest > 0.5) & (mask_np > 0.5) & (pseudo_np <= 0.5)
        mask_np[area_interest_tn_mask] = 0.0
        pseudo_np = (pseudo_np > 0.5) | (mask_np < 0.5)
        pseudo_np = pseudo_np.astype(np.float32)
        connected_pseudo = torch.tensor(pseudo_np, device=pseudo.device).unsqueeze(0).unsqueeze(0).to(dtype=pseudo.dtype)
        connected_mask = torch.tensor(mask_np, device=mask.device).unsqueeze(0).unsqueeze(0).to(dtype=mask.dtype)
        return (connected_pseudo, connected_mask)

    def _predict_with_zoom_augs(self, model, x):
        zoom_factors = [1.0, 1.0, 1.75, 2.0]
        device = x.device
        aug_inputs = []
        wrapped_model = WrappedModelNoSigmoid(model)
        for zoom_factor in zoom_factors:
            if zoom_factor == 1.0:
                aug_inputs.append(x)
            else:
                new_size = (int(x.shape[2] * zoom_factor), int(x.shape[3] * zoom_factor))
                x_zoomed = nn.functional.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
                aug_inputs.append(x_zoomed)
        aug_outputs = []
        for zoom_factor, aug_input in zip(zoom_factors, aug_inputs):
            assert zoom_factor >= 1.0
            if zoom_factor == 1.0:
                aug_outputs.append(model(aug_input)[0])
            else:
                pred_this = sliding_window_inference(aug_input, (x.shape[2], x.shape[3]), sw_batch_size=3, predictor=wrapped_model)
                pred = F.interpolate(pred_this, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                aug_outputs.append(pred)
        aug_outputs = torch.cat(aug_outputs, dim=0)
        return aug_outputs

    def _do_train_tt_sfuda(self, model, original_model, gnn_model, x, gnn_info, epoch, name, frangi=None):
        device = self.device
        self.optimizer.zero_grad()
        self.optimizer_gnn.zero_grad()
        gnn_conf = self.kwargs['gnn']
        if gnn_conf.get('delete_edge_weight_under', 0) > 0:
            for gnn_info_idx in range(len(gnn_info)):
                gnn_info[gnn_info_idx] = self._do_gnn_delete_edge_weight_under(gnn_info[gnn_info_idx], gnn_conf['delete_edge_weight_under'])
        criterion = cross_entropy_dice_2d
        if self.kwargs.get('test_time_aug_type') == 'flip':
            aug_input = build_pseduo_augmentation_plpb_2d(x.cpu().numpy())
        elif self.kwargs.get('test_time_aug_type') in [None, 'orig']:
            aug_input = build_pseduo_augmentation_2d(x.cpu().numpy())
        elif self.kwargs.get('test_time_aug_type') == 'flip_only':
            aug_input = build_pseudo_augmentation_flip_2d(x)
        elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray':
            aug_input = build_pseudo_augmentation_flip_with_gray_2d(x)
        elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray_v2':
            aug_input = build_pseudo_augmentation_flip_with_gray_2d_v2(x)
        elif self.kwargs.get('test_time_aug_type') == 'zoom':
            aug_input = torch.clone(x).unsqueeze(1)
        else:
            raise ValueError('Invalid test_time_aug_type')
        loss_dic = {}
        with temporary_eval(original_model, self.kwargs.get('pseudo_label_original_model_eval', False)), torch.no_grad():
            if self.kwargs['map_contrastive_loss']['use'] and self.kwargs['map_contrastive_loss'].get('use_model_uncert_for_moco', False):
                also_make_model_aug_output = True
            else:
                also_make_model_aug_output = False
            ps_outputs = []
            mask_outputs = []
            weighted_ents = []
            for j in range(aug_input.shape[0]):
                if self.kwargs.get('test_time_aug_type') == 'zoom':
                    aug_output = self._predict_with_zoom_augs(original_model, aug_input[j].to(device))
                else:
                    aug_output, leak_dic = original_model(aug_input[j].to(device))
                    if also_make_model_aug_output:
                        aug_output_model, _ = model(aug_input[j].to(device))
                if self.kwargs.get('test_time_aug_type') == 'flip':
                    aug_output = flip_back_plpb_2d(aug_output)
                    if also_make_model_aug_output:
                        aug_output_model = flip_back_plpb_2d(aug_output_model)
                elif self.kwargs.get('test_time_aug_type') in ['flip_only', 'flip_with_gray']:
                    aug_output = flip_back_2d_flip(aug_output)
                    if also_make_model_aug_output:
                        aug_output_model = flip_back_2d_flip(aug_output_model)
                elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray_v2':
                    aug_output = flip_back_2d_flip_with_gray_v2(aug_output)
                    if also_make_model_aug_output:
                        aug_output_model = flip_back_2d_flip_with_gray_v2(aug_output_model)
                if self.kwargs.get('voting_method', 'tt') == 'tt':
                    ps_output, mask_output, weighted_ent = uncert_voting_2d(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True))
                    if also_make_model_aug_output:
                        ps_output_model, mask_output_model, weighted_ent_model = uncert_voting_2d(aug_output_model.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True))
                elif self.kwargs.get('voting_method', 'tt') == 'pseudo':
                    ps_output, mask_output, weighted_ent = uncert_voting_2d_use_pseudo(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True), thres=self.kwargs.get('tt_pseudo_thres', 0.1))
                    if also_make_model_aug_output:
                        ps_output_model, mask_output_model, weighted_ent_model = uncert_voting_2d_use_pseudo(aug_output_model.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True), thres=self.kwargs.get('tt_pseudo_thres', 0.1))
                if also_make_model_aug_output:
                    mask_output_model = 1 - mask_output_model
                mask_output = 1 - mask_output
                ps_outputs.append(ps_output)
                mask_outputs.append(mask_output)
                weighted_ents.append(weighted_ent)
            ps_outputs = torch.cat(ps_outputs, dim=0).to(device=device, dtype=torch.float32)
            mask_outputs = torch.cat(mask_outputs, dim=0).to(device=device, dtype=torch.float32)
            weighted_ents = torch.cat(weighted_ents, dim=0).to(device=device, dtype=torch.float32)
            if self.kwargs.get('connect_pseudo_mask_endpoints') and (not (self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] and self.kwargs['frangi_in_pseudo'].get('hard', False))):
                ps_outputs, mask_outputs = self._connect_pseudo_mask_endpoints(ps_outputs, mask_outputs)
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
                is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
                frangi = frangi.to(device)[0]
                mask_outputs_this = ((mask_outputs > 0.5) | (ps_outputs > 0.5)).to(torch.float32)
                if self.kwargs.get('force_true_certain', False):
                    mask_outputs = mask_outputs_this
                if self.kwargs['frangi_in_pseudo'].get('add_to_area_interest', False) is False:
                    ps_outputs = self._mix_frangi_and_pseudo(frangi, ps_outputs, mask_outputs_this, hard_frangi=is_hard_frangi, force_pseudo_true=self.kwargs['frangi_in_pseudo'].get('force_pseudo_true', False))
                else:
                    ps_outputs, mask_outputs = self._mix_frangi_and_pseudo_add_area_interest(frangi, ps_outputs, mask_outputs_this, hard_frangi=is_hard_frangi, force_pseudo_true=self.kwargs['frangi_in_pseudo'].get('force_pseudo_true', False))
                if is_hard_frangi and self.kwargs.get('connect_pseudo_mask_endpoints'):
                    ps_outputs, mask_outputs = self._connect_pseudo_mask_endpoints(ps_outputs, mask_outputs)
        self.optimizer.zero_grad()
        assert aug_input.shape[0] == 1
        aug_input = aug_input[0]
        for i in range(aug_input.shape[0]):
            aug_input_i = aug_input[i].to(device).unsqueeze(0)
            output_i, _ = model(aug_input_i)
            if self.kwargs.get('test_time_aug_type') == 'flip':
                ps_output_i = rotate_2D_torch(ps_outputs.clone(), i)
                mask_output_i = rotate_2D_torch(mask_outputs.clone(), i, background_color=1)
            elif self.kwargs.get('test_time_aug_type') in ['flip_only', 'flip_with_gray']:
                ps_output_i = rotate_and_flip_2D_torch(ps_outputs.clone(), i)
                mask_output_i = rotate_and_flip_2D_torch(mask_outputs.clone(), i, background_color=1)
            elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray_v2':
                if i < 7:
                    ps_output_i = rotate_and_flip_2D_torch(ps_outputs.clone(), i)
                    mask_output_i = rotate_and_flip_2D_torch(mask_outputs.clone(), i, background_color=1)
                else:
                    ps_output_i = ps_outputs.clone()
                    mask_output_i = mask_outputs.clone()
            else:
                ps_output_i = ps_outputs
                mask_output_i = mask_outputs
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] and self.kwargs['frangi_in_pseudo'].get('two_part_loss', True):
                smoothl1_no_reduction = nn.SmoothL1Loss(reduction='none')
                seg_loss_1 = masked_bce_with_logits_dice(output_i, ps_output_i.to(device=device, dtype=torch.float32), mask_output_i.to(device=device, dtype=torch.float32), uncertain_weight=self.kwargs.get('uncertain_weight', 0.0), uncertain_weight_dice=self.kwargs.get('uncertain_weight_dice', 0.0))
                mask_uncertain_and_positive = 1 - mask_output_i
                if not is_hard_frangi:
                    seg_loss_2 = smoothl1_no_reduction(torch.sigmoid(output_i) * mask_uncertain_and_positive, ps_output_i.to(device) * mask_uncertain_and_positive)
                else:
                    seg_loss_2 = F.binary_cross_entropy_with_logits(output_i, ps_output_i.to(device=device, dtype=torch.float32), reduction='none') * mask_uncertain_and_positive
                seg_loss = seg_loss_1 + seg_loss_2.sum() / (mask_uncertain_and_positive.sum() + 1e-05)
                if is_hard_frangi and self.kwargs['frangi_in_pseudo'].get('hard_frangi_dice', 0.0) > 0.0:
                    seg_loss += WeightedDiceLoss(sigmoid=False)(torch.sigmoid(output_i) * mask_uncertain_and_positive, ps_output_i.to(device=device, dtype=torch.float32) * mask_uncertain_and_positive) * self.kwargs['frangi_in_pseudo']['hard_frangi_dice']
            else:
                seg_loss = masked_bce_with_logits_dice(output_i, ps_output_i.to(device=device, dtype=torch.float32), mask_output_i.to(device=device, dtype=torch.float32) if self.kwargs.get('use_uncertainty_mask', True) else torch.ones_like(ps_output_i), uncertain_weight=self.kwargs.get('uncertain_weight', 0.0), uncertain_weight_dice=self.kwargs.get('uncertain_weight_dice', 0.0))
            ent_loss = sigmoid_entropy_loss(output_i)
            loss = self.kwargs['loss_weights']['seg'] * seg_loss + self.kwargs['loss_weights']['ent'] * ent_loss
            if self.kwargs.get('cldice') and self.kwargs['cldice']['use'] and (epoch >= self.kwargs['cldice'].get('begin_epoch', 0)):
                certain_positive_region = ps_output_i * mask_output_i
                uncertain_positive_region = ps_output_i * (1 - mask_output_i)
                uncertain_negative_region = (1 - ps_output_i) * (1 - mask_output_i)
                combined_mask_for_cldice = certain_positive_region + self.kwargs['cldice']['beta1'] * uncertain_positive_region + self.kwargs['cldice']['beta2'] * uncertain_negative_region
                if self.kwargs['cldice'].get('type', 'orig') == 'orig':
                    cldice_loss_func = SoftClDice()
                elif self.kwargs['cldice']['type'] == 'weighted':
                    cldice_loss_func = SoftClDiceWeighted(weight_type=self.kwargs['cldice'].get('weight_type', 'exp'), power=self.kwargs['cldice'].get('power', 0.1))
                elif self.kwargs['cldice']['type'] == 'cbdice':
                    cldice_loss_func = SoftcbDiceLoss(smooth=0.001)
                if self.kwargs['cldice'].get('type', 'orig') in ['orig', 'weighted']:
                    cldice = cldice_loss_func(combined_mask_for_cldice.to(device=device, dtype=torch.float32), torch.sigmoid(output_i))
                else:
                    cldice = cldice_loss_func(torch.cat([-output_i, output_i], dim=1), combined_mask_for_cldice.to(device=device, dtype=torch.float32))
                loss = loss + self.kwargs['cldice']['weight'] * cldice
            if loss.item() > 999999:
                print('Warning: loss is too large, skip this batch')
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
        loss_dic['seg_loss'] = seg_loss.item()
        loss_dic['ent_loss'] = ent_loss.item()
        aug_input_gnn = aug_input[0].to(device).unsqueeze(0)
        pred, leak_dic = model(aug_input_gnn)
        pred = torch.sigmoid(pred)
        pred = torch.cat([1 - pred, pred], dim=1)
        with temporary_eval(original_model, self.kwargs.get('pseudo_label_original_model_eval', False)):
            if self.kwargs.get('map_contrastive_loss') and self.kwargs['map_contrastive_loss']['use'] and (epoch >= self.kwargs['map_contrastive_loss'].get('begin_epoch', 0)):
                if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                    if self.kwargs['map_contrastive_loss'].get('orig_model_for_no_aug', False):
                        with torch.no_grad():
                            pred_orig_model, leak_dic_orig_model = original_model(aug_input_gnn)
                            pred_orig_model = torch.sigmoid(pred_orig_model)
                            pred_orig_model = torch.cat([1 - pred_orig_model, pred_orig_model], dim=1)
                    aug_pred_conf = self.kwargs['map_contrastive_loss'].get('aug_pred_type', {'type': 'gray'})
                    aug_pred_type = aug_pred_conf.get('type', 'gray')
                    if aug_pred_type == 'follow':
                        r = random.randint(1, aug_input.shape[0] - 1)
                        aug_input_1 = aug_input[r].to(device).unsqueeze(0)
                        pred_1, leak_dic_1 = model(aug_input_1)
                        pred_1 = torch.sigmoid(pred_1)
                        pred_1 = torch.cat([1 - pred_1, pred_1], dim=1)
                        if self.kwargs.get('test_time_aug_type') == 'flip':
                            pred_1 = rotate_2D_torch(pred_1, 8 - r)
                            for key in leak_dic_1.keys():
                                leak_dic_1[key] = rotate_2D_torch(leak_dic_1[key], 8 - r)
                        elif self.kwargs.get('test_time_aug_type') in ['flip_only', 'flip_with_gray', 'flip_with_gray_v2']:
                            if 1 <= r <= 3:
                                code = 4 - r
                            elif r == 7:
                                code = 0
                            else:
                                code = r
                            pred_1 = rotate_and_flip_2D_torch(pred_1, code)
                            for key in leak_dic_1.keys():
                                leak_dic_1[key] = rotate_and_flip_2D_torch(leak_dic_1[key], code)
                    else:
                        if aug_pred_type == 'gray':
                            aug_class = tfs.Grayscale(num_output_channels=3)
                        elif aug_pred_type == 'tt_sfuda':
                            aug_class = StrongAugmentation()
                        elif aug_pred_type == 'bezier':
                            aug_class = Bezier(flip=False, prob=1.0, num_free_coordinates=aug_pred_conf.get('num_free_coordinates', 1), max_rnd_val=aug_pred_conf.get('max_rnd_val', 0.35), min_rnd_val=aug_pred_conf.get('min_rnd_val', 0.15))
                        elif aug_pred_type == 'randaug':
                            aug_class = RandAug(aug_pred_conf['n_augs'])
                        elif aug_pred_type == 'blur':
                            aug_class = RandomGaussianBlur(p=1)
                        elif aug_pred_type == 'contrast':
                            aug_class = RandomContrastStretch(p=1)
                        aug_class = aug_class.to(device)
                        aug_input_1 = aug_class(x)
                        pred_1, leak_dic_1 = model(aug_input_1)
                        pred_1 = torch.sigmoid(pred_1)
                        pred_1 = torch.cat([1 - pred_1, pred_1], dim=1)
                map_contrastive_loss_conf = self.kwargs['map_contrastive_loss']
                crop_size = (96, 96)
                if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                    assert map_contrastive_loss_conf['kl_type'] in ['moco', 'l1']
                    if map_contrastive_loss_conf['kl_type'] == 'moco':
                        if type(map_contrastive_loss_conf['map_from']) == str:
                            map_contrastive_loss_conf['map_from'] = [map_contrastive_loss_conf['map_from']]
                        if not map_contrastive_loss_conf.get('moco_no_uncertain', False):
                            contrastive_loss_func = AugRandomContrastiveLossMoco2d(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=map_contrastive_loss_conf.get('uncertain_prob', 0), wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss=map_contrastive_loss_conf.get('aug_l1_loss', {'use': False}), moco_weight=map_contrastive_loss_conf.get('moco_weight', 1.0))
                            contrastive_loss = 0
                            for map_from in map_contrastive_loss_conf['map_from']:
                                feature_to_refine = leak_dic[map_from] if not self.kwargs['map_contrastive_loss'].get('orig_model_for_no_aug', False) else leak_dic_orig_model[map_from]
                                if map_contrastive_loss_conf.get('orig_feature_detach', False):
                                    feature_to_refine = feature_to_refine.detach()
                                if map_contrastive_loss_conf.get('use_model_uncert_for_moco', False):
                                    contrastive_loss_this = contrastive_loss_func(feature_to_refine, leak_dic_1[map_from], ps_output_model, mask_output_model.to(dtype=torch.float32), entropy_map=weighted_ent_model[0])
                                else:
                                    contrastive_loss_this = contrastive_loss_func(feature_to_refine, leak_dic_1[map_from], ps_output, mask_output.to(dtype=torch.float32), entropy_map=weighted_ent[0]) * map_contrastive_loss_conf['weight']
                                contrastive_loss = contrastive_loss + contrastive_loss_this
                        else:
                            assert map_contrastive_loss_conf.get('aug_l1_loss', {'use': False})['use'] is True, 'When moco_no_uncertain is True, aug_l1_loss must be used'
                            contrastive_loss_func_l1 = AugRandomContrastiveLossMoco2d(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=map_contrastive_loss_conf.get('uncertain_prob', 0), wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss=map_contrastive_loss_conf.get('aug_l1_loss', {'use': False}), moco_weight=0.0)
                            contrastive_loss_func_moco = AugRandomContrastiveLossMoco2d(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=0, wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss={'use': False}, moco_weight=map_contrastive_loss_conf.get('moco_weight', 1.0))
                            contrastive_loss = 0
                            for map_from in map_contrastive_loss_conf['map_from']:
                                feature_to_refine = leak_dic[map_from] if not self.kwargs['map_contrastive_loss'].get('orig_model_for_no_aug', False) else leak_dic_orig_model[map_from]
                                if map_contrastive_loss_conf.get('orig_feature_detach', False):
                                    feature_to_refine = feature_to_refine.detach()
                                if map_contrastive_loss_conf.get('use_model_uncert_for_moco', False):
                                    contrastive_loss_this_l1 = contrastive_loss_func_l1(feature_to_refine, leak_dic_1[map_from], ps_output_model, mask_output_model.to(dtype=torch.float32), entropy_map=weighted_ent_model[0])
                                    contrastive_loss_this_moco = contrastive_loss_func_moco(feature_to_refine, leak_dic_1[map_from], ps_output_model, mask_output_model.to(dtype=torch.float32), entropy_map=weighted_ent_model[0])
                                else:
                                    contrastive_loss_this_l1 = contrastive_loss_func_l1(feature_to_refine, leak_dic_1[map_from], ps_output, mask_output.to(dtype=torch.float32), entropy_map=weighted_ent[0])
                                    contrastive_loss_this_moco = contrastive_loss_func_moco(feature_to_refine, leak_dic_1[map_from], ps_output, mask_output.to(dtype=torch.float32), entropy_map=weighted_ent[0])
                                contrastive_loss_this = (contrastive_loss_this_l1 + contrastive_loss_this_moco) * map_contrastive_loss_conf['weight']
                                contrastive_loss = contrastive_loss + contrastive_loss_this
                    elif map_contrastive_loss_conf['kl_type'] == 'l1':
                        contrastive_loss = 0
                        for map_from in map_contrastive_loss_conf['map_from']:
                            feature_to_refine = leak_dic[map_from] if not self.kwargs['map_contrastive_loss'].get('orig_model_for_no_aug', False) else leak_dic_orig_model[map_from]
                            if map_contrastive_loss_conf.get('orig_feature_detach', False):
                                feature_to_refine = feature_to_refine.detach()
                            contrastive_loss_this = feature_map_consistency_loss(feature_to_refine, leak_dic_1[map_from], weighted_ent[0], weight_map_scale=map_contrastive_loss_conf.get('weight_map_scale', 1.0)) * map_contrastive_loss_conf['weight']
                            contrastive_loss = contrastive_loss + contrastive_loss_this
                    else:
                        raise ValueError('Invalid kl_type for map_contrastive_loss')
                loss_dic['map_contr_loss'] = contrastive_loss.item()
                contrastive_loss.backward(retain_graph=True)
        if gnn_conf['weight'] > 0:
            if gnn_conf.get('pred_edge', False):
                loss_gnn, loss_dic, gnn_dic_list = self._calc_loss_gnn_pred_edges(leak_dic, gnn_model, gnn_info, gnn_conf, x, ps_outputs, mask_outputs, loss_dic, epoch=epoch)
            else:
                if gnn_conf.get('anchor_model_as_uncertainty_begin_epoch', -1) >= 0:
                    pred_anchor, leak_dic_anchor = self.anchor_model(x)
                    pred_stack = torch.cat([torch.logit(pred[:, 1:2], eps=1e-06), pred_anchor], dim=0)
                    _, mask_outputs, _ = uncert_voting_2d_use_pseudo(pred_stack.detach(), and_fn=True)
                aug_pred_lst = []
                if len(gnn_conf.get('aug_pred_types', [])) > 0:
                    for aug_pred_type in gnn_conf['aug_pred_types']:
                        assert aug_pred_type in ['gray', 'tt_sfuda', 'bezier', 'randaug', 'follow']
                        if aug_pred_type == 'gray':
                            aug_class = tfs.Grayscale(num_output_channels=3)
                        elif aug_pred_type == 'tt_sfuda':
                            aug_class = StrongAugmentation()
                        elif aug_pred_type == 'bezier':
                            aug_class = Bezier(flip=False, prob=1.0, num_free_coordinates=1, max_rnd_val=0.85, min_rnd_val=0.75)
                        elif aug_pred_type == 'randaug':
                            aug_class = RandAug(3)
                        aug_class = aug_class.to(device)
                        aug_input_1 = aug_class(x)
                        pred_1, leak_dic_1 = model(aug_input_1)
                        aug_pred_lst.append(leak_dic_1[gnn_conf['map_from']].detach())
                loss_gnn, loss_dic, gnn_dic_list = self._calc_loss_gnn_weak_supervision(leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_outputs, mask_outputs, loss_dic, aug_pred_lst, epoch=epoch, name=name[0])
            loss_gnn.backward(retain_graph=True)
        else:
            loss_gnn = torch.tensor(0.0)
        self.optimizer_gnn.step()
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            self.optimizer_refiner.step()
            self.optimizer_refiner.zero_grad()
        loss_dic['gnn_loss'] = loss_gnn.item()
        if self.kwargs.get('segmentor') and self.kwargs['segmentor'].get('use'):
            feature_map = prepare_feature_map_for_segmentor_2d(x, feature_map_from=self.kwargs['segmentor']['feature_map_from'], leak_dic=leak_dic, gnn_dic_list=gnn_dic_list, device=self.device, mode=self.kwargs['segmentor'].get('merge_mode', 'add'), scales=self.kwargs['segmentor'].get('scales', '12'))
            loss_segmentor = self._do_train_segmentor(leak_dic, feature_map, ps_outputs, mask_outputs)
            loss_dic['loss_segmentor'] = loss_segmentor
        self.optimizer.step()
        return loss_dic

    def _do_train_segmentor(self, encoder_features, feature_map, ps_output, mask_output):
        if self.kwargs['segmentor']['feature_map_from'] == 'map_0':
            segmentor_pred = self.segmentor(feature_map, [encoder_features['skip_3'], encoder_features['skip_2'], encoder_features['skip_1'], encoder_features['skip_0']], x_map_num=0)
        elif self.kwargs['segmentor']['feature_map_from'] == 'map_1':
            segmentor_pred = self.segmentor(feature_map, [encoder_features['skip_2'], encoder_features['skip_1'], encoder_features['skip_0']], x_map_num=1)
        elif self.kwargs['segmentor']['feature_map_from'] == 'map_2':
            segmentor_pred = self.segmentor(feature_map, [encoder_features['skip_1'], encoder_features['skip_0']], x_map_num=2)
        elif self.kwargs['segmentor']['feature_map_from'] == 'map_3':
            segmentor_pred = self.segmentor(feature_map, [encoder_features['skip_0']], x_map_num=3)
        elif self.kwargs['segmentor']['feature_map_from'] == 'map_4':
            segmentor_pred = self.segmentor(feature_map, [], x_map_num=4)
        segmentor_pred = torch.sigmoid(segmentor_pred)
        segmentor_dice_loss = masked_bce_with_logits_dice(segmentor_pred, ps_output.to(device=self.device, dtype=torch.float32), mask=mask_output.to(device=self.device, dtype=torch.float32)) * self.kwargs['segmentor'].get('seg_loss_weight', 1.0)
        segmentor_ent_loss = entropy_loss_no_sigmoid(segmentor_pred) * self.kwargs['segmentor'].get('ent_loss_weight', 1.0)
        segmentor_loss = segmentor_dice_loss + segmentor_ent_loss
        self.optimizer_segmentor.zero_grad()
        segmentor_loss.backward()
        self.optimizer_segmentor.step()
        return segmentor_loss.item()

    def _evaluate_gnn_single(self, interps, pred_gnn, pred_gnn_binary):
        gnn_acc = (pred_gnn_binary == interps['gt']).to(dtype=pred_gnn.dtype).mean().item()
        gnn_precision = ((pred_gnn_binary * interps['gt']).sum() / (pred_gnn_binary.sum() + 1e-05)).item()
        gnn_recall = ((pred_gnn_binary * interps['gt']).sum() / (interps['gt'].sum() + 1e-05)).item()
        try:
            gnn_auc = roc_auc_score(interps['gt'].cpu().detach().numpy(), pred_gnn.cpu().detach().numpy())
        except ValueError:
            gnn_auc = -1
        dic = {'gnn_acc': gnn_acc, 'gnn_precision': gnn_precision, 'gnn_recall': gnn_recall, 'gnn_auc': gnn_auc}
        return dic

    def _evaluate_gnn(self, interps, pred, pred_gnn_1, pred_gnn_2, pred_gnn_binary_1, pred_gnn_binary_2):
        gnn_acc = (pred_gnn_binary_1 == interps['gt'][0]).to(dtype=pred.dtype).mean().item()
        gnn_acc_2 = (pred_gnn_binary_2 == interps['gt'][1]).to(dtype=pred.dtype).mean().item()
        gnn_precision = ((pred_gnn_binary_1 * interps['gt'][0]).sum() / (pred_gnn_binary_1.sum() + 1e-05)).item()
        gnn_precision_2 = ((pred_gnn_binary_2 * interps['gt'][1]).sum() / (pred_gnn_binary_2.sum() + 1e-05)).item()
        gnn_recall = ((pred_gnn_binary_1 * interps['gt'][0]).sum() / (interps['gt'][0].sum() + 1e-05)).item()
        gnn_recall_2 = ((pred_gnn_binary_2 * interps['gt'][1]).sum() / (interps['gt'][1].sum() + 1e-05)).item()

        def calc_auc(pred, gt):
            try:
                auc = roc_auc_score(gt.cpu().detach().numpy(), pred.cpu().detach().numpy())
            except ValueError:
                auc = -1
            return auc
        gnn_auc_1 = calc_auc(pred_gnn_1, interps['gt'][0])
        gnn_auc_2 = calc_auc(pred_gnn_2, interps['gt'][1])
        dic = {'gnn_acc': gnn_acc, 'gnn_precision': gnn_precision, 'gnn_recall': gnn_recall, 'gnn_auc': gnn_auc_1, 'gnn_acc_2': gnn_acc_2, 'gnn_precision_2': gnn_precision_2, 'gnn_recall_2': gnn_recall_2, 'gnn_auc_2': gnn_auc_2}
        return dic

    def _pred_edge_from_vertices(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def _calc_loss_gnn_pred_edges(self, leak_dic, gnn_model, gnn_info, gnn_conf, x, ps_output, mask_output, loss_dic, epoch=None):
        feature = leak_dic[gnn_conf['map_from']][0].unsqueeze(0)
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            feature = self.refiner(feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
        if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
            feature = nn.Upsample(size=x.shape[2:], mode='bilinear')(feature)
        if gnn_conf.get('scales', 1) == 1:
            assert len(gnn_info) == 1
            gnn_info = gnn_info[0]
            if gnn_conf.get('use_slic', False):
                batch_size_slic = 10000
                gnn_info['segments'] = gnn_info['segments'].to(device=self.device)
                gnn_info['coords'] = torch.flip(gnn_info['coords'], dims=[2])
                assert gnn_info['segments'].max().item() == len(gnn_info['coords'][0])
                feature_interp = region_aggregation(feature, gnn_info['segments'], len(gnn_info['coords'][0]), method=gnn_conf.get('slic_method', 'mean'))[0]
                gt_interp = region_aggregation(ps_output.to(dtype=torch.float32), gnn_info['segments'], len(gnn_info['coords'][0]), method='mean')[0].squeeze(-1)
                uncertainty_interp = region_aggregation(mask_output.to(dtype=torch.float32), gnn_info['segments'], len(gnn_info['coords'][0]), method='mean')[0].squeeze(-1)
                gt_interp = threshold_torch(gt_interp, 0.2)
                uncertainty_interp = threshold_torch(uncertainty_interp, 0.5)
            else:
                interp_mode = gnn_conf.get('interp_mode', 'bilinear')
                feature_interp = sample_in_feature_map_2d_torch(feature, gnn_info['coords'], mode=interp_mode)[0]
                gt_interp = sample_in_feature_map_2d_torch(ps_output.to(dtype=torch.float32), gnn_info['coords'], mode='nearest')[0].squeeze(-1)
                uncertainty_interp = sample_in_feature_map_2d_torch(mask_output.to(dtype=torch.float32), gnn_info['coords'], mode='nearest')[0].squeeze(-1)
            data = Data(x=feature_interp, edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_attr=gnn_info['edge_weights'][0].to(device=self.device) if self.kwargs['gnn'].get('use_edge_attr', True) else None)
            pos_edge_index = data.edge_index
            neg_edge_index = sample_negative_edges(edge_index=data.edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1) * gnn_conf.get('neg_pos_ratio', 1))
            if gnn_model.__class__.__name__ in ['GAT', 'GCN', 'GATUNet', 'GraphUNet', 'GNNWithRoPE']:
                if gnn_model.__class__.__name__ in ['GAT']:
                    pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_attr=gnn_info['edge_weights'][0].to(device=self.device) if self.kwargs['gnn'].get('use_edge_attr', True) else None)
                elif gnn_model.__class__.__name__ in ['GraphUNet', 'GATUNet', 'GCN']:
                    pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_weight=gnn_info['edge_weights'][0].to(device=self.device))
                elif gnn_model.__class__.__name__ in ['GNNWithRoPE']:
                    pred_gnn = gnn_model(feature_interp, positions=gnn_info['coords'][0].to(device=self.device, dtype=torch.float32), edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_weight=gnn_info['edge_weights'][0].to(device=self.device))
                pos_scores = self._pred_edge_from_vertices(pred_gnn, pos_edge_index)
                pos_labels = torch.ones(pos_scores.size(0), device=pos_scores.device)
                neg_scores = self._pred_edge_from_vertices(pred_gnn, neg_edge_index)
                neg_labels = torch.zeros(neg_scores.size(0), device=neg_scores.device)
                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)
                loss = F.binary_cross_entropy_with_logits(scores, labels) * gnn_conf.get('weight', 1.0)
                loss_dic['gnn_edge_pred_loss'] = loss.item()
                probs = scores.sigmoid().cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                pred_auc = roc_auc_score(labels, probs)
                pred_ap = average_precision_score(labels, probs)
                loss_dic['gnn_edge_pred_auc'] = pred_auc
                loss_dic['gnn_edge_pred_ap'] = pred_ap
        return (loss, loss_dic, [])

    def _calc_pos_neg_labels_gnn(self, dic):
        l = len(dic['pred_gnn_1'])
        return_dic = {}
        for k in dic.keys():
            stacked_pred = torch.stack(dic[k], axis=0)
            stacked_pred = torch.stack([1 - stacked_pred, stacked_pred], axis=1)
            stacked_pred_argmax = (stacked_pred > 0.5).to(torch.float32)
            stacked_pred_argmin = 1 - stacked_pred_argmax
            pos_labels = torch.max(stacked_pred_argmax, dim=0)[0]
            neg_labels = torch.sum(stacked_pred_argmin, dim=0)
            return_dic[k] = [pos_labels, neg_labels]
        return return_dic

    def _calc_weight_weak_supervision_gnn(self, Y, probs):
        weight = Y * probs
        sum_weight = weight.sum(dim=1).unsqueeze(1).repeat(1, weight.shape[1], 1)
        weight = torch.div(weight, sum_weight).detach_()
        return weight

    def _calc_symmetric_loss(self, outputs, partial_weight, com_weight):

        def probs_weighted_ce(probs, weight):
            out_log = torch.log(probs + 1e-08)
            loss = -(out_log * weight).sum(dim=1)
            return loss.mean()
        probs = F.softmax(outputs, dim=1)
        partial_loss = probs_weighted_ce(probs, partial_weight)
        com_loss = probs_weighted_ce(1 - probs, com_weight)
        return partial_loss + com_loss

    def _calc_loss_gnn_weak_supervision(self, leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic, aug_pred_lst=[], epoch=None, name=None, is_test=False):
        is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
        feature = leak_dic[gnn_conf['map_from']][0].unsqueeze(0)
        smoothl1_no_reduction = nn.SmoothL1Loss(reduction='none')
        smoothl1_loss = nn.SmoothL1Loss()
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            feature = self.refiner(feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
        if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
            feature = nn.Upsample(size=x.shape[2:], mode='bilinear')(feature)
        if gnn_conf.get('scales', 1) == 1:
            assert len(gnn_info) == 1
            gnn_info = gnn_info[0]
            if gnn_conf.get('use_slic', False):
                batch_size_slic = 10000
                gnn_info['segments'] = gnn_info['segments'].to(device=self.device)
                gnn_info['coords'] = torch.flip(gnn_info['coords'], dims=[2])
                assert gnn_info['segments'].max().item() == len(gnn_info['coords'][0])
                feature_interp = region_aggregation(feature, gnn_info['segments'], len(gnn_info['coords'][0]), method=gnn_conf.get('slic_method', 'mean'))[0]
                gt_interp = region_aggregation(ps_output.to(dtype=torch.float32), gnn_info['segments'], len(gnn_info['coords'][0]), method='mean')[0].squeeze(-1)
                uncertainty_interp = region_aggregation(mask_output.to(dtype=torch.float32), gnn_info['segments'], len(gnn_info['coords'][0]), method='mean')[0].squeeze(-1)
                gt_interp = threshold_torch(gt_interp, 0.2)
                uncertainty_interp = threshold_torch(uncertainty_interp, 0.5)
            else:
                interp_mode = gnn_conf.get('interp_mode', 'bilinear')
                feature_interp = sample_in_feature_map_2d_torch(feature, gnn_info['coords'], mode=interp_mode)[0]
                gt_interp = sample_in_feature_map_2d_torch(ps_output.to(dtype=torch.float32), gnn_info['coords'], mode='nearest')[0].squeeze(-1)
                uncertainty_interp = sample_in_feature_map_2d_torch(mask_output.to(dtype=torch.float32), gnn_info['coords'], mode='nearest')[0].squeeze(-1)
            if gnn_conf.get('use_original_gt', False):
                gt_interp = gnn_info['gt'][0].to(device=self.device)
                uncertainty_interp = gnn_info['uncertainty'][0].to(device=self.device)

            def do_gnn_forward(feature_interp, gnn_model, gnn_info, gnn_conf):
                if gnn_model.__class__.__name__ in ['GAT']:
                    pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_attr=gnn_info['edge_weights'][0].to(device=self.device) if self.kwargs['gnn'].get('use_edge_attr', True) else None)
                elif gnn_model.__class__.__name__ in ['GraphUNet', 'GATUNet', 'GCN']:
                    pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_weight=gnn_info['edge_weights'][0].to(device=self.device))
                elif gnn_model.__class__.__name__ in ['GNNWithRoPE']:
                    pred_gnn = gnn_model(feature_interp, positions=gnn_info['coords'][0].to(device=self.device, dtype=torch.float32), edge_index=gnn_info['edges'][0].T.to(device=self.device, dtype=torch.int64), edge_weight=gnn_info['edge_weights'][0].to(device=self.device))
                return pred_gnn
            pred_gnn = do_gnn_forward(feature_interp, gnn_model, gnn_info, gnn_conf)
            if len(aug_pred_lst) > 0:
                loss_aug_pred = 0
                pred_gnn_augs_lst = []
                for idx in range(len(aug_pred_lst)):
                    aug_pred_feature = aug_pred_lst[idx]
                    if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
                        aug_pred_feature = self.refiner(aug_pred_feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
                    if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
                        aug_pred_feature = nn.Upsample(size=x.shape[2:], mode='bilinear')(aug_pred_feature)
                    if gnn_conf.get('use_slic', False):
                        feature_aug = region_aggregation(aug_pred_feature, gnn_info['segments'], len(gnn_info['coords'][0]), method=gnn_conf.get('slic_method', 'mean'))[0]
                    else:
                        feature_aug = sample_in_feature_map_2d_torch(aug_pred_feature, gnn_info['coords'].squeeze(0), mode=interp_mode)[0]
                    pred_gnn_aug = do_gnn_forward(feature_aug, gnn_model, gnn_info, gnn_conf)
                    pred_gnn_augs_lst.append(pred_gnn_aug)
                    loss_aug_pred = loss_aug_pred + F.smooth_l1_loss(torch.sigmoid(pred_gnn), torch.sigmoid(pred_gnn_aug))
                    aug_pred_ce_weight = gnn_conf.get('aug_pred_ce_weight', 0.0)
                    if aug_pred_ce_weight > 0.0:
                        loss_aug_pred = loss_aug_pred + aug_pred_ce_weight * bceloss0(torch.sigmoid(pred_gnn_aug).squeeze(-1), gt_interp).mean()
                loss_aug_pred = loss_aug_pred * gnn_conf.get('aug_pred_weight', 0.0)
                if gnn_conf.get('uncertainty_by_aug_pred', False):
                    stacked_preds = torch.stack([torch.sigmoid(pred_gnn)] + [torch.sigmoid(p) for p in pred_gnn_augs_lst], dim=0)
                    uncertainty_interp = 1 - threshold_torch(stacked_preds.std(dim=0).squeeze(-1), 0.08)
                    loss_dic['uncertainty_perc'] = uncertainty_interp.mean().item()
            pred_gnn = torch.sigmoid(pred_gnn).squeeze(-1)
            loss_type = gnn_conf.get('loss_type', 'focal')
            if gnn_conf.get('elr') and gnn_conf['elr']['use'] and (not is_test):
                assert gnn_conf.get('gnn_memory_bank') is None or not gnn_conf['gnn_memory_bank']['use'], 'ELR and GNN memory bank cannot be used together.'
                elr_zeta = gnn_conf['elr'].get('elr_lambda', 0.8)
                self.gnn_memory_bank.initialize_gt(name, gt_interp)
                self.gnn_memory_bank.update_gt_ema(name, pred_gnn.detach(), elr_zeta)
                bceloss0 = ELRLossOneClass(reduction='none')
            elif loss_type == 'focal':
                bceloss0 = BinaryFocalLoss(neg_weight=1, gamma=2, reduction='none')
            elif loss_type == 'dice':
                bceloss0 = DiceLoss1d()
            elif loss_type == 'bce':
                bceloss0 = BCELossPosWeight(pos_weight=1, reduction='none')
            elif loss_type == 'gce':
                bceloss0 = SymmetricGCE(q=gnn_conf.get('gce_q', 0.7), alpha=gnn_conf.get('gce_alpha', 0), reduction='none')
            loss_gnn = bceloss0(pred_gnn, gt_interp)
            if gnn_conf.get('gnn_memory_bank') and gnn_conf['gnn_memory_bank']['use'] and (not is_test):
                if gnn_conf.get('mod_gt', False):
                    self.gnn_memory_bank.initialize_gt(name, gt_interp)
                if epoch % gnn_conf['gnn_memory_bank'].get('update_interval', 1) == 0:
                    self.gnn_memory_bank.add_predictions(name, pred_gnn.detach())
                if epoch > gnn_conf['gnn_memory_bank'].get('start_epoch', gnn_conf['gnn_memory_bank']['max_memory_epoch'] * gnn_conf['gnn_memory_bank'].get('update_interval', 1)):
                    mask_new_1 = calc_agreement_from_memory_bank(self.gnn_memory_bank.storage, name) > gnn_conf['gnn_memory_bank'].get('agreement_threshold', 0.5)
                    uncertainty_interp = mask_new_1.to(dtype=uncertainty_interp.dtype)
                    if gnn_conf.get('mod_gt', False):
                        gt_last_1 = self.gnn_memory_bank.get_gt(name)
                        gt_this_1 = mask_new_1 * (pred_gnn_1.detach() > 0.5).to(dtype=gt_interp.dtype) + (1 - mask_new_1) * gt_last_1.to(dtype=gt_interp.dtype)
                        gt_interp = gt_this_1
                        self.gnn_memory_bank.update_gt(name, gt_this_1)
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] and gnn_conf.get('use_frangi', True):
                if is_hard_frangi:
                    loss_gnn_l1 = bceloss0(pred_gnn, gt_interp)
                else:
                    loss_gnn_l1 = smoothl1_no_reduction(pred_gnn, gt_interp)
                loss_gnn_l1 = safe_mean_torch(loss_gnn_l1, uncertainty_interp < 0.5) * gnn_conf['weight']
                loss_gnn = safe_mean_torch(loss_gnn, uncertainty_interp > 0.5) * gnn_conf['weight']
                loss_gnn = loss_gnn + loss_gnn_l1_1
            elif gnn_conf.get('uncertain_loss_weight', -1) >= 0:
                loss_gnn_certain = safe_mean_torch(loss_gnn, uncertainty_interp > 0.5) * torch.sum(uncertainty_interp > 0.5) / uncertainty_interp.shape[0] * gnn_conf['weight']
                loss_gnn_uncertain = safe_mean_torch(loss_gnn, uncertainty_interp <= 0.5) * torch.sum(uncertainty_interp <= 0.5) / uncertainty_interp.shape[0] * gnn_conf['weight'] * gnn_conf['uncertain_loss_weight']
                loss_gnn = loss_gnn_certain + loss_gnn_uncertain
            else:
                loss_gnn = loss_gnn.mean() * gnn_conf['weight']
            if gnn_conf.get('ent_weight', 0):
                ent_loss = entropy_loss_no_sigmoid(pred_gnn) * gnn_conf['ent_weight']
                loss_gnn = loss_gnn + ent_loss
                loss_dic['gnn_ent_loss'] = ent_loss.item()
            if len(aug_pred_lst) > 0:
                loss_gnn = loss_gnn + loss_aug_pred
                loss_dic['gnn_aug_pred_loss'] = loss_aug_pred.item()
            pred_gnn_binary = (pred_gnn > 0.5).to(dtype=pred.dtype)
            dic_gnn_results = self._evaluate_gnn_single({'gt': gt_interp}, pred_gnn, pred_gnn_binary)
            loss_dic.update(dic_gnn_results)
            if gnn_conf.get('graph_smooth_loss') and epoch >= gnn_conf.get('graph_smooth_start_epoch', 0):
                smooth_1 = graph_smoothness_loss(pred_gnn, gnn_info[0]['edges'][0].T)
                loss_gnn = loss_gnn + smooth_1 * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
                loss_dic['graph_smooth_loss'] = smooth_1.item() * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
            if gnn_conf.get('consistency') and epoch >= gnn_conf.get('consistency_start_epoch', 0):
                if gnn_conf.get('use_slic', False):
                    pred_interp = region_aggregation(pred[:, 1:2], gnn_info['segments'], len(gnn_info['coords'][0]), method='mean')[0]
                else:
                    pred_interp = sample_in_feature_map_2d_torch(pred[:, 1:2], gnn_info['coords'], mode='bilinear')[0]
                pred_interp = pred_interp.squeeze(-1)
                if gnn_conf.get('consistency_gnn_detach', False):
                    pred_gnn = pred_gnn.detach()
                consistency_loss = smoothl1_loss(pred_interp, pred_gnn) * gnn_conf['consistency']
                loss_gnn = loss_gnn + consistency_loss
                loss_dic['consistency_loss'] = consistency_loss.item()
            gnn_dic_list = []
            for i in range(1):
                gnn_dic_list.append({'coords': gnn_info['coords'].squeeze(0).cpu().numpy(), 'edges': gnn_info['edges'].squeeze(0).cpu().numpy(), 'edge_weights': gnn_info['edge_weights'].squeeze(0).cpu().numpy(), 'pred': [pred_gnn.detach().cpu().numpy()][i], 'gt': gt_interp.detach().cpu().numpy(), 'pred_2d': pred.squeeze(0).detach().cpu().numpy()})
        elif gnn_conf.get('scales', 1) == 2:
            interps = {}
            interp_mode = gnn_conf.get('interp_mode', 'bilinear')
            for idx, gnn_info_this_scale in enumerate(gnn_info):
                if gnn_conf.get('use_slic', False):
                    batch_size_slic = 10000
                    gnn_info[idx]['coords'] = torch.flip(gnn_info[idx]['coords'], dims=[2])
                    feature_interp = region_aggregation(feature, gnn_info_this_scale['segments'], len(gnn_info_this_scale['coords'][0]), method=gnn_conf.get('slic_method', 'mean'))[0]
                    gt_interp = region_aggregation(ps_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['segments'], len(gnn_info_this_scale['coords'][0]), method='mean')[0].squeeze(-1)
                    uncertainty_interp = region_aggregation(mask_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['segments'], len(gnn_info_this_scale['coords'][0]), method='mean')[0].squeeze(-1)
                    gt_interp = threshold_torch(gt_interp, 0.2)
                    uncertainty_interp = threshold_torch(uncertainty_interp, 0.5)
                else:
                    feature_interp = sample_in_feature_map_2d_torch(feature, gnn_info_this_scale['coords'], mode=interp_mode)[0]
                    gt_interp = sample_in_feature_map_2d_torch(ps_output.to(dtype=torch.float32), gnn_info_this_scale['coords'], mode='nearest')[0].squeeze(-1)
                    uncertainty_interp = sample_in_feature_map_2d_torch(mask_output.to(dtype=torch.float32), gnn_info_this_scale['coords'], mode='nearest')[0].squeeze(-1)
                if gnn_conf.get('use_original_gt', False):
                    gt_interp = gnn_info_this_scale['gt'][0].to(device=self.device, dtype=torch.float32)
                    uncertainty_interp = gnn_info_this_scale['uncertainty'][0].to(device=self.device, dtype=torch.float32)
                interps = dict_element_lst_append(interps, {'feature': feature_interp, 'gt': gt_interp, 'uncertainty': uncertainty_interp})
            loss_type = gnn_conf.get('loss_type', 'focal')
            if loss_type == 'focal':
                bceloss0 = BinaryFocalLoss(neg_weight=1, gamma=2, reduction='none')
            elif loss_type == 'dice':
                bceloss0 = DiceLoss1d()
            elif loss_type == 'bce':
                bceloss0 = BCELossPosWeight(pos_weight=1, reduction='none')
            if gnn_conf.get('aug_edges_num', 0) > 0:
                aug_edges_lst_0 = []
                aug_edges_lst_1 = []
                aug_edge_weights_lst_0 = []
                aug_edge_weights_lst_1 = []
                edge_length_0 = len(gnn_info[0]['edge_weights'].squeeze(0))
                full_idxs_0 = torch.arange(0, edge_length_0, dtype=torch.int64)
                edge_length_1 = len(gnn_info[1]['edge_weights'].squeeze(0))
                full_idxs_1 = torch.arange(0, edge_length_1, dtype=torch.int64)
                for _ in range(gnn_conf['aug_edges_num']):
                    aug_ratio = gnn_conf.get('aug_edges_ratio', 0.8)
                    num_aug_0 = int(edge_length_0 * aug_ratio)
                    num_aug_1 = int(edge_length_1 * aug_ratio)
                    aug_idxs_0 = full_idxs_0[torch.randperm(edge_length_0)[:num_aug_0]]
                    aug_idxs_1 = full_idxs_1[torch.randperm(edge_length_1)[:num_aug_1]]
                    aug_edges_lst_0.append(gnn_info[0]['edges'][:, aug_idxs_0])
                    aug_edge_weights_lst_0.append(gnn_info[0]['edge_weights'][:, aug_idxs_0])
                    aug_edges_lst_1.append(gnn_info[1]['edges'][:, aug_idxs_1])
                    aug_edge_weights_lst_1.append(gnn_info[1]['edge_weights'][:, aug_idxs_1])
            if gnn_model.__class__.__name__ == 'TwoScaleGAT':
                raise NotImplementedError('TwoScaleGAT is not implemented for weak supervision')
            elif gnn_model.__class__.__name__ in ['TwoScaleGATInterp', 'MGGNN']:
                pred_gnn_1, pred_gnn_2 = gnn_model(feature1=interps['feature'][0], edge_index1=gnn_info[0]['edges'].squeeze(0).T, edge_attr1=gnn_info[0]['edge_weights'].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos1=gnn_info[0]['coords'].squeeze(0), feature2=interps['feature'][1], edge_index2=gnn_info[1]['edges'].squeeze(0).T, edge_attr2=gnn_info[1]['edge_weights'].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos2=gnn_info[1]['coords'].squeeze(0))
                if gnn_conf.get('aug_edges_num', 0) > 0:
                    aug_edges_results_dic = {}
                    with torch.no_grad():
                        for aug_idx in range(gnn_conf['aug_edges_num']):
                            pred_gnn_1_aug, pred_gnn_2_aug = gnn_model(feature1=interps['feature'][0], edge_index1=aug_edges_lst_0[aug_idx].squeeze(0).T, edge_attr1=aug_edge_weights_lst_0[aug_idx].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos1=gnn_info[0]['coords'].squeeze(0), feature2=interps['feature'][1], edge_index2=aug_edges_lst_1[aug_idx].squeeze(0).T, edge_attr2=aug_edge_weights_lst_1[aug_idx].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos2=gnn_info[1]['coords'].squeeze(0))
                            aug_edges_results_dic = dict_element_lst_append(aug_edges_results_dic, {'pred_gnn_1': torch.sigmoid(pred_gnn_1_aug).squeeze(-1), 'pred_gnn_2': torch.sigmoid(pred_gnn_2_aug).squeeze(-1)})
                    aug_labels_dic = self._calc_pos_neg_labels_gnn(aug_edges_results_dic)
                    logits_gnn_1 = torch.stack([1 - pred_gnn_1, pred_gnn_1], dim=0).unsqueeze(0).squeeze(-1)
                    logits_gnn_2 = torch.stack([1 - pred_gnn_2, pred_gnn_2], dim=0).unsqueeze(0).squeeze(-1)
                    probs_gnn_1 = F.softmax(logits_gnn_1, dim=1).detach()
                    probs_gnn_2 = F.softmax(logits_gnn_2, dim=1).detach()
                    pos_weight_gnn_1 = self._calc_weight_weak_supervision_gnn(aug_labels_dic['pred_gnn_1'][0].unsqueeze(0), probs_gnn_1)
                    neg_weight_gnn_1 = self._calc_weight_weak_supervision_gnn(aug_labels_dic['pred_gnn_1'][1].unsqueeze(0), probs_gnn_1)
                    pos_weight_gnn_2 = self._calc_weight_weak_supervision_gnn(aug_labels_dic['pred_gnn_2'][0].unsqueeze(0), probs_gnn_2)
                    neg_weight_gnn_2 = self._calc_weight_weak_supervision_gnn(aug_labels_dic['pred_gnn_2'][1].unsqueeze(0), probs_gnn_2)
                    loss_aug_edges_1 = self._calc_symmetric_loss(logits_gnn_1, pos_weight_gnn_1, neg_weight_gnn_1)
                    loss_aug_edges_2 = self._calc_symmetric_loss(logits_gnn_2, pos_weight_gnn_2, neg_weight_gnn_2)
                    loss_aug_edges = (loss_aug_edges_1 + loss_aug_edges_2) * gnn_conf.get('aug_edges_weight', 0.0)
                if len(aug_pred_lst) > 0:
                    loss_aug_pred = 0
                    for idx in range(len(aug_pred_lst)):
                        aug_pred_feature = aug_pred_lst[idx]
                        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
                            aug_pred_feature = self.refiner(aug_pred_feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
                        if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
                            aug_pred_feature = nn.Upsample(size=x.shape[2:], mode='bilinear')(aug_pred_feature)
                        feature_aug_1 = sample_in_feature_map_2d_torch(aug_pred_feature, gnn_info[0]['coords'].squeeze(0), mode=interp_mode)[0]
                        feature_aug_2 = sample_in_feature_map_2d_torch(aug_pred_feature, gnn_info[1]['coords'].squeeze(0), mode=interp_mode)[0]
                        pred_gnn_1_aug, pred_gnn_2_aug = gnn_model(feature1=feature_aug_1, edge_index1=gnn_info[0]['edges'].squeeze(0).T, edge_attr1=gnn_info[0]['edge_weights'].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos1=gnn_info[0]['coords'].squeeze(0), feature2=feature_aug_2, edge_index2=gnn_info[1]['edges'].squeeze(0).T, edge_attr2=gnn_info[1]['edge_weights'].squeeze(0) if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos2=gnn_info[1]['coords'].squeeze(0))
                        loss_aug_pred = loss_aug_pred + F.smooth_l1_loss(torch.sigmoid(pred_gnn_1), torch.sigmoid(pred_gnn_1_aug))
                        loss_aug_pred = loss_aug_pred + F.smooth_l1_loss(torch.sigmoid(pred_gnn_2), torch.sigmoid(pred_gnn_2_aug))
                        aug_pred_ce_weight = gnn_conf.get('aug_pred_ce_weight', 0.0)
                        if aug_pred_ce_weight > 0.0:
                            loss_aug_pred = loss_aug_pred + aug_pred_ce_weight * (bceloss0(torch.sigmoid(pred_gnn_1_aug).squeeze(-1), interps['gt'][0]).mean() + bceloss0(torch.sigmoid(pred_gnn_2_aug).squeeze(-1), interps['gt'][1]).mean())
                    loss_aug_pred = loss_aug_pred * gnn_conf.get('aug_pred_weight', 0.0)
            else:
                raise NotImplementedError('Only GAT is supported for multi-scale')
            pred_gnn_1 = torch.sigmoid(pred_gnn_1).squeeze(-1)
            pred_gnn_2 = torch.sigmoid(pred_gnn_2).squeeze(-1)
            if gnn_conf.get('gnn_memory_bank') and gnn_conf['gnn_memory_bank']['use']:
                if gnn_conf.get('mod_gt', False):
                    self.gnn_memory_bank.initialize_gt(name, interps['gt'][0])
                    self.gnn_memory_bank_2.initialize_gt(name, interps['gt'][1])
                if epoch % gnn_conf['gnn_memory_bank'].get('update_interval', 1) == 0:
                    self.gnn_memory_bank.add_predictions(name, pred_gnn_1.detach())
                    self.gnn_memory_bank_2.add_predictions(name, pred_gnn_2.detach())
                if epoch > gnn_conf['gnn_memory_bank'].get('start_epoch', gnn_conf['gnn_memory_bank']['max_memory_epoch'] * gnn_conf['gnn_memory_bank'].get('update_interval', 1)):
                    mask_new_1 = calc_agreement_from_memory_bank(self.gnn_memory_bank.storage, name) > gnn_conf['gnn_memory_bank'].get('agreement_threshold', 0.5)
                    mask_new_2 = calc_agreement_from_memory_bank(self.gnn_memory_bank_2.storage, name) > gnn_conf['gnn_memory_bank'].get('agreement_threshold', 0.5)
                    interps['uncertainty'][0] = mask_new_1.to(dtype=interps['uncertainty'][0].dtype)
                    interps['uncertainty'][1] = mask_new_2.to(dtype=interps['uncertainty'][1].dtype)
                    if gnn_conf.get('mod_gt', False):
                        gt_last_1 = self.gnn_memory_bank.get_gt(name)
                        gt_last_2 = self.gnn_memory_bank_2.get_gt(name)
                        gt_this_1 = mask_new_1 * (pred_gnn_1.detach() > 0.5).to(dtype=interps['gt'][0].dtype) + (1 - mask_new_1) * gt_last_1.to(dtype=interps['gt'][0].dtype)
                        gt_this_2 = mask_new_2 * (pred_gnn_2.detach() > 0.5).to(dtype=interps['gt'][1].dtype) + (1 - mask_new_2) * gt_last_2.to(dtype=interps['gt'][1].dtype)
                        interps['gt'][0] = gt_this_1
                        interps['gt'][1] = gt_this_2
                        self.gnn_memory_bank.update_gt(name, gt_this_1)
                        self.gnn_memory_bank_2.update_gt(name, gt_this_2)
            loss_gnn_1 = bceloss0(pred_gnn_1, interps['gt'][0])
            loss_gnn_2 = bceloss0(pred_gnn_2, interps['gt'][1])
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] and gnn_conf.get('use_frangi', True):
                if is_hard_frangi:
                    loss_gnn_l1_1 = bceloss0(pred_gnn_1, interps['gt'][0])
                    loss_gnn_l1_2 = bceloss0(pred_gnn_2, interps['gt'][1])
                else:
                    loss_gnn_l1_1 = smoothl1_no_reduction(pred_gnn_1, interps['gt'][0])
                    loss_gnn_l1_2 = smoothl1_no_reduction(pred_gnn_2, interps['gt'][1])
                loss_gnn_l1_1 = safe_mean_torch(loss_gnn_l1_1, interps['uncertainty'][0] < 0.5) * gnn_conf['weight']
                loss_gnn_l1_2 = safe_mean_torch(loss_gnn_l1_2, interps['uncertainty'][1] < 0.5) * gnn_conf['weight']
                loss_gnn_1 = safe_mean_torch(loss_gnn_1, interps['uncertainty'][0] > 0.5) * gnn_conf['weight']
                loss_gnn_2 = safe_mean_torch(loss_gnn_2, interps['uncertainty'][1] > 0.5) * gnn_conf['weight']
                loss_gnn = loss_gnn_1 + loss_gnn_2 + loss_gnn_l1_1 + loss_gnn_l1_2
            elif gnn_conf.get('uncertain_loss_weight', -1) >= 0:
                loss_gnn_1_certain = safe_mean_torch(loss_gnn_1, interps['uncertainty'][0] > 0.5) * torch.sum(interps['uncertainty'][0] > 0.5) / interps['uncertainty'][0].shape[0] * gnn_conf['weight']
                loss_gnn_2_certain = safe_mean_torch(loss_gnn_2, interps['uncertainty'][1] > 0.5) * torch.sum(interps['uncertainty'][0] > 0.5) / interps['uncertainty'][0].shape[0] * gnn_conf['weight']
                loss_gnn_1_uncertain = safe_mean_torch(loss_gnn_1, interps['uncertainty'][0] <= 0.5) * torch.sum(interps['uncertainty'][0] <= 0.5) / interps['uncertainty'][0].shape[0] * gnn_conf['weight'] * gnn_conf['uncertain_loss_weight']
                loss_gnn_2_uncertain = safe_mean_torch(loss_gnn_2, interps['uncertainty'][1] <= 0.5) * torch.sum(interps['uncertainty'][0] <= 0.5) / interps['uncertainty'][0].shape[0] * gnn_conf['weight'] * gnn_conf['uncertain_loss_weight']
                loss_gnn = loss_gnn_1_certain + loss_gnn_2_certain + loss_gnn_1_uncertain + loss_gnn_2_uncertain
            else:
                loss_gnn = loss_gnn_1.mean() * gnn_conf['weight'] + loss_gnn_2.mean() * gnn_conf['weight']
            if gnn_conf.get('ent_weight', 0):
                ent_loss_1 = entropy_loss_no_sigmoid(pred_gnn_1) * gnn_conf['ent_weight']
                ent_loss_2 = entropy_loss_no_sigmoid(pred_gnn_2) * gnn_conf['ent_weight']
                ent_loss = ent_loss_1 + ent_loss_2
                loss_gnn = loss_gnn + ent_loss
                loss_dic['gnn_ent_loss'] = ent_loss.item()
            if gnn_conf.get('aug_edges_num', 0) > 0:
                loss_gnn = loss_gnn + loss_aug_edges
                loss_dic['gnn_aug_edges_loss'] = loss_aug_edges.item()
            if len(aug_pred_lst) > 0:
                loss_gnn = loss_gnn + loss_aug_pred
                loss_dic['gnn_aug_pred_loss'] = loss_aug_pred.item()
            pred_gnn_binary_1 = (pred_gnn_1 > 0.5).to(dtype=pred.dtype)
            pred_gnn_binary_2 = (pred_gnn_2 > 0.5).to(dtype=pred.dtype)
            dic_gnn_results = self._evaluate_gnn(interps, pred, pred_gnn_1, pred_gnn_2, pred_gnn_binary_1, pred_gnn_binary_2)
            loss_dic.update(dic_gnn_results)
            if gnn_conf.get('graph_smooth_loss') and epoch >= gnn_conf.get('graph_smooth_start_epoch', 0):
                smooth_1 = graph_smoothness_loss(pred_gnn_1, gnn_info[0]['edges'][0].T)
                smooth_2 = graph_smoothness_loss(pred_gnn_2, gnn_info[1]['edges'][0].T)
                loss_gnn = loss_gnn + (smooth_1 + smooth_2) * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
                loss_dic['graph_smooth_loss'] = (smooth_1 + smooth_2).item() * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
            if gnn_conf.get('consistency') and epoch >= gnn_conf.get('consistency_start_epoch', 0):
                if gnn_conf.get('use_slic', False):
                    pred_interp_1 = region_aggregation(pred[:, 1:2], gnn_info[0]['segments'], len(gnn_info[0]['coords'][0]), method='mean')[0].squeeze(-1)
                    pred_interp_2 = region_aggregation(pred[:, 1:2], gnn_info[1]['segments'], len(gnn_info[1]['coords'][0]), method='mean')[0].squeeze(-1)
                else:
                    pred_interp_1 = sample_in_feature_map_2d_torch(pred[:, 1:2], gnn_info[0]['coords'], mode='bilinear')[0].squeeze(-1)
                    pred_interp_2 = sample_in_feature_map_2d_torch(pred[:, 1:2], gnn_info[1]['coords'], mode='bilinear')[0].squeeze(-1)
                if gnn_conf.get('consistency_gnn_detach', False):
                    pred_gnn_1 = pred_gnn_1.detach()
                    pred_gnn_2 = pred_gnn_2.detach()
                if gnn_conf.get('consistency_no_sigmoid', False):
                    pred_interp_1 = torch.logit(pred_interp_1, eps=1e-06)
                    pred_interp_2 = torch.logit(pred_interp_2, eps=1e-06)
                    pred_gnn_1 = torch.logit(pred_gnn_1, eps=1e-06)
                    pred_gnn_2 = torch.logit(pred_gnn_2, eps=1e-06)
                if gnn_conf.get('trunk_connection'):
                    if gnn_conf['trunk_connection'].get('weighted_dt', False):
                        dt_to_trunk = component_weighted_distance_transform_torch(threshold_torch(pred[:, 1:2], 0.5).to(dtype=torch.float32))
                    else:
                        if gnn_conf['trunk_connection'].get('min_component_size', 0) > 0:
                            pred_binary = torch.from_numpy(remove_small_objects((pred[0][1] > 0.5).detach().cpu().numpy().astype(np.bool_), gnn_conf['trunk_connection']['min_component_size'])).unsqueeze(0).unsqueeze(0)
                        else:
                            pred_binary = threshold_torch(pred[:, 1:2], 0.5)
                        dt_to_trunk = -distance_transform_edt_torch(pred_binary.to(dtype=torch.float32)) * (1 - pred_binary.to(dtype=torch.float32))
                    dt_to_trunk_interp = sample_in_feature_map_2d_torch(dt_to_trunk, gnn_info[0]['coords'].squeeze(0), mode='bilinear')[0].squeeze(-1)
                    dt_to_trunk_interp = torch.clamp(dt_to_trunk_interp, min=0, max=50)
                    weight = torch.zeros_like(dt_to_trunk_interp, device=dt_to_trunk_interp.device)
                    weight[dt_to_trunk_interp == 0] = 1
                    beta = gnn_conf['trunk_connection'].get('beta', 1.0)
                    init_weight = gnn_conf['trunk_connection'].get('init_weight', 1.0)
                    distance_thres_1 = gnn_conf['trunk_connection'].get('distance_thres_1', 3.0)
                    weight[(dt_to_trunk_interp > 0) & (dt_to_trunk_interp <= distance_thres_1)] = init_weight
                    if gnn_conf['trunk_connection'].get('weight_scheme', 'inverse') == 'inverse':
                        weight[dt_to_trunk_interp > distance_thres_1] = 1 / dt_to_trunk_interp[dt_to_trunk_interp > distance_thres_1] ** beta * init_weight
                    elif gnn_conf['trunk_connection'].get('weight_scheme', 'inverse') == 'exp':
                        weight[dt_to_trunk_interp > distance_thres_1] = torch.exp(-beta * dt_to_trunk_interp[dt_to_trunk_interp > distance_thres_1]) * init_weight
                    weight[(pred_interp_1 > 0.5) & (pred_gnn_1 < 0.5)] = 0
                    if gnn_conf['trunk_connection'].get('normalize', True):
                        weight = weight / weight.mean()
                    weight = weight.to(device=pred_interp_1.device, dtype=pred_interp_1.dtype)
                    consistency_loss = smoothl1_no_reduction(pred_interp_1, pred_gnn_1) * weight * gnn_conf['consistency']
                    if gnn_conf['trunk_connection'].get('pred_penalty', 0) > 0:
                        weight_penalty = dt_to_trunk_interp / distance_thres_1
                        weight_penalty = torch.clamp(weight_penalty, max=1.0)
                        if gnn_conf.get('consistency_no_sigmoid', False):
                            pred_interp_1_sigmoid = torch.sigmoid(pred_interp_1)
                        else:
                            pred_interp_1_sigmoid = pred_interp_1
                        consistency_loss = consistency_loss + pred_interp_1_sigmoid * weight_penalty * gnn_conf['trunk_connection']['pred_penalty'] * gnn_conf['consistency']
                    consistency_loss = consistency_loss.mean()
                elif gnn_conf.get('dt_weight'):
                    assert not gnn_conf.get('trunk_connection')
                    dt_map = distance_transform_edt_torch((ps_output > 0.5).to(dtype=torch.float32))
                    dt_interp = sample_in_feature_map_2d_torch(dt_map, gnn_info[0]['coords'], mode='bilinear')[0].squeeze(-1)
                    dt_local_maxima = get_local_maxima_grid(gnn_info[0]['coords'][0].to(dt_interp.device), dt_map.squeeze(0).squeeze(0), d=3, H=ps_output.shape[-2], W=ps_output.shape[-1])
                    weight_0_mask = ((dt_interp < 0) & (dt_local_maxima > 0)).to(dtype=torch.float32)
                    weight = torch.zeros_like(dt_interp)
                    weight[dt_local_maxima < 0] = 1.0
                    thin_vessel_init_weight = gnn_conf['dt_weight'].get('thin_vessel_weight_init', 4.0)
                    weight[(dt_local_maxima > 0) & (dt_local_maxima < gnn_conf['dt_weight'].get('distance_thres_1', 5.0))] = thin_vessel_init_weight - (thin_vessel_init_weight - 1) / gnn_conf['dt_weight'].get('distance_thres_1', 5.0) * dt_local_maxima[(dt_local_maxima > 0) & (dt_local_maxima < gnn_conf['dt_weight'].get('distance_thres_1', 5.0))]
                    weight[dt_local_maxima >= gnn_conf['dt_weight'].get('distance_thres_1', 5.0)] = gnn_conf['dt_weight'].get('thick_vessel_weight_init', 0.3) * torch.exp(-gnn_conf['dt_weight'].get('thick_vessel_weight_exp', 1) * (dt_local_maxima[dt_local_maxima >= gnn_conf['dt_weight'].get('distance_thres_1', 5.0)] - gnn_conf['dt_weight'].get('distance_thres_1', 5.0)))
                    if gnn_conf['dt_weight'].get('use_weight_0_mask', True):
                        weight = weight * (1 - weight_0_mask)
                    if gnn_conf['dt_weight'].get('normalize', True):
                        weight = weight / weight.mean()
                    weight = weight.to(device=pred_interp_1.device, dtype=pred_interp_1.dtype)
                    if gnn_conf['dt_weight'].get('ablation'):
                        consistency_loss = smoothl1_no_reduction(pred_interp_1, gnn_info[0]['gt'][0].to(device=self.device) * 14 - 7) * gnn_conf['consistency'] * weight
                    else:
                        consistency_loss = smoothl1_no_reduction(pred_interp_1, pred_gnn_1) * weight * gnn_conf['consistency']
                    consistency_loss = consistency_loss.mean()
                else:
                    consistency_loss = smoothl1_loss(pred_interp_1, pred_gnn_1) * gnn_conf['consistency']
                loss_gnn = loss_gnn + consistency_loss * gnn_conf['weight']
                loss_dic['consistency_loss'] = consistency_loss.item() * gnn_conf['weight']
            gnn_dic_list = []
            for i in range(len(gnn_info)):
                gnn_dic_list.append({'coords': gnn_info[i]['coords'].squeeze(0).cpu().numpy(), 'edges': gnn_info[i]['edges'].squeeze(0).cpu().numpy(), 'edge_weights': gnn_info[i]['edge_weights'].squeeze(0).cpu().numpy(), 'pred': [pred_gnn_1.detach().cpu().numpy(), pred_gnn_2.detach().cpu().numpy()][i], 'gt': interps['gt'][i].detach().cpu().numpy(), 'pred_2d': pred.squeeze(0).detach().cpu().numpy()})
        return (loss_gnn, loss_dic, gnn_dic_list)

    def _calc_pseudo_mask_online(self, x, orig_model):
        x_ori = x.detach().cpu()
        pred_lst = []
        features_lst = []
        for i in range(8):
            with torch.no_grad():
                x_aug = torch.clone(x_ori).to(device=self.device, dtype=torch.float32)
                flip_code = i % 8
                x_aug = rotate_2D_torch(x_aug, flip_code)
                pred, leak_dic = orig_model(x_aug)
                pred = rotate_2D_torch(pred, 8 - flip_code)
                pred = torch.softmax(pred, dim=1)
                pred = pred.detach().cpu()
            pred_lst.append(pred)
            feature = leak_dic['last_feature_map'].detach().cpu()
            feature = flip_3D_torch(feature, 8 - flip_code)
            features_lst.append(feature)
        pred_lst = torch.stack(pred_lst)
        pred = pred_lst.mean(dim=0)
        std = pred_lst.std(dim=0)
        max_pos = pred_lst[:, :, 1:2].max(dim=0)[0]
        max_pos = torch.cat([1 - max_pos, max_pos], dim=1)
        write_nii(std[:, 1:2], 'std.nii.gz')
        feature = torch.stack(features_lst).mean(dim=0)
        mask, pseudo_label = get_mask_from_uncertainty_2d(pred, std, feature)
        return (mask, pseudo_label)

    def _wrap_model_forward_eval(self, **kwargs):
        if kwargs.get('model_to_test'):
            return WrappedModel(self.models_dict[kwargs['model_to_test']])
        else:
            return WrappedModel(self.model)

    def test_gnn(self, test_loader, use_pseudo_as_gt=True, epoch=None):
        model = self.model
        original_model = self.orig_model
        gnn_model = self.gnn
        device = self.device
        model.eval()
        gnn_model.eval()
        metrics_dic = {}
        with torch.no_grad():
            for batch in tqdm(test_loader):
                x = batch['image'].to(device=device, dtype=torch.float32)
                label = batch['label'].to(device=device, dtype=torch.float32)
                gnn_dic = batch['gnn']
                gnn2_dic = batch.get('gnn2', None)
                frangi = batch['frangi'].to(device=device, dtype=torch.float32) if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] else None
                gnn_info = [gnn_dic, gnn2_dic] if gnn2_dic is not None else [gnn_dic]
                device = self.device
                criterion = cross_entropy_dice_2d
                if self.kwargs.get('test_time_aug_type') == 'flip':
                    aug_input = build_pseduo_augmentation_plpb_2d(x.cpu().numpy())
                elif self.kwargs.get('test_time_aug_type') in [None, 'orig']:
                    aug_input = build_pseduo_augmentation_2d(x.cpu().numpy())
                elif self.kwargs.get('test_time_aug_type') == 'flip_only':
                    aug_input = build_pseudo_augmentation_flip_2d(x)
                elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray':
                    aug_input = build_pseudo_augmentation_flip_with_gray_2d(x)
                elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray_v2':
                    aug_input = build_pseudo_augmentation_flip_with_gray_2d_v2(x)
                elif self.kwargs.get('test_time_aug_type') == 'zoom':
                    aug_input = torch.clone(x).unsqueeze(1)
                else:
                    raise ValueError('Invalid test_time_aug_type')
                loss_dic = {}
                with temporary_eval(original_model, self.kwargs.get('pseudo_label_original_model_eval', False)), torch.no_grad():
                    ps_outputs = []
                    mask_outputs = []
                    weighted_ents = []
                    for j in range(aug_input.shape[0]):
                        if self.kwargs.get('test_time_aug_type') == 'zoom':
                            aug_output = self._predict_with_zoom_augs(original_model, aug_input[j].to(device))
                        else:
                            aug_output, leak_dic = original_model(aug_input[j].to(device))
                        if self.kwargs.get('test_time_aug_type') == 'flip':
                            aug_output = flip_back_plpb_2d(aug_output)
                        elif self.kwargs.get('test_time_aug_type') in ['flip_only', 'flip_with_gray']:
                            aug_output = flip_back_2d_flip(aug_output)
                        elif self.kwargs.get('test_time_aug_type') == 'flip_with_gray_v2':
                            aug_output = flip_back_2d_flip_with_gray_v2(aug_output)
                        if self.kwargs.get('voting_method', 'tt') == 'tt':
                            ps_output, mask_output, weighted_ent = uncert_voting_2d(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True), uncertain_to_positive=self.kwargs.get('tt_uncertain_to_positive', True))
                        elif self.kwargs.get('voting_method', 'tt') == 'pseudo':
                            ps_output, mask_output, weighted_ent = uncert_voting_2d_use_pseudo(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True), thres=self.kwargs.get('tt_pseudo_thres', 0.1), uncertain_to_positive=self.kwargs.get('tt_uncertain_to_positive', True))
                        mask_output = 1 - mask_output
                        ps_outputs.append(ps_output)
                        mask_outputs.append(mask_output)
                        weighted_ents.append(weighted_ent)
                    ps_outputs = torch.cat(ps_outputs, dim=0).to(device=device, dtype=torch.float32)
                    mask_outputs = torch.cat(mask_outputs, dim=0).to(device=device, dtype=torch.float32)
                    weighted_ents = torch.cat(weighted_ents, dim=0).to(device=device, dtype=torch.float32)
                    if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
                        is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
                        frangi = frangi.to(device)[0]
                        mask_outputs_this = ((mask_outputs > 0.5) | (ps_outputs > 0.5)).to(torch.float32)
                        if self.kwargs.get('force_true_certain', False):
                            mask_outputs = mask_outputs_this
                        ps_outputs = self._mix_frangi_and_pseudo(frangi, ps_outputs, mask_outputs_this, hard_frangi=is_hard_frangi)
                with torch.no_grad():
                    aug_input_gnn = aug_input[0][0].to(device).unsqueeze(0)
                    pred, leak_dic = model(aug_input_gnn)
                    pred = torch.sigmoid(pred)
                    pred = torch.cat([1 - pred, pred], dim=1)
                if self.kwargs['gnn'].get('pred_edge', False):
                    if use_pseudo_as_gt:
                        _, metrics_dic_this, _ = self._calc_loss_gnn_pred_edges(leak_dic, gnn_model, gnn_info, self.kwargs['gnn'], x, ps_outputs, mask_outputs, {}, epoch=epoch)
                    else:
                        _, metrics_dic_this, _ = self._calc_loss_gnn_pred_edges(leak_dic, gnn_model, gnn_info, self.kwargs['gnn'], x, label, torch.ones_like(label), {}, epoch=epoch)
                elif use_pseudo_as_gt:
                    _, metrics_dic_this, gnn_dic_lst = self._calc_loss_gnn_weak_supervision(leak_dic, gnn_model, gnn_info, self.kwargs['gnn'], x, pred, ps_outputs, mask_outputs, {}, epoch=epoch, is_test=True)
                else:
                    _, metrics_dic_this, gnn_dic_lst = self._calc_loss_gnn_weak_supervision(leak_dic, gnn_model, gnn_info, self.kwargs['gnn'], x, pred, label, torch.ones_like(label), {}, epoch=epoch, is_test=True)
                metrics_dic = dict_element_lst_append(metrics_dic, metrics_dic_this)
                if not self.kwargs['gnn'].get('pred_edge', False):
                    for num, gnn_dic in enumerate(gnn_dic_lst):
                        epoch_string = '' if epoch is None else f'_epoch_{epoch}'
                        self._save_gnn_prediction(gnn_dic, f'gnn_level_{num + 1}{epoch_string}', batch['name'][0], (ps_outputs[0][0] > 0.5).to(torch.float32).cpu().numpy() if use_pseudo_as_gt else label[0][0].cpu().numpy(), use_pseudo_as_gt=use_pseudo_as_gt)
            for k in metrics_dic.keys():
                metrics_dic[k] = np.mean(metrics_dic[k])
            self.logger.write_dict(metrics_dic)

    def _save_gnn_prediction(self, gnn_dic, fd_name, name, pred_2d, use_pseudo_as_gt=True):
        save_fd = define_fd(osp.join(self.base_folder, fd_name + ('_pseudo' if use_pseudo_as_gt else '')))
        save_path = osp.join(save_fd, name + '.png')
        fig, ax = visualize_segmentation_with_points(pred_2d, gnn_dic['coords'], (gnn_dic['pred'] > 0.5).astype(np.float32), gnn_dic['gt'])
        save_path_pkl = osp.join(save_fd, name + '.pkl')
        with open(save_path_pkl, 'wb') as f:
            gnn_dic['pred_2d'] = pred_2d
            pickle.dump(gnn_dic, f)
        plt.savefig(save_path)
        plt.cla()
        plt.close(fig)

    def test_segmentor(self, loader, loader_name='Test', model_path=None, thres=0.5):
        self._set_eval()
        model_to_test = 'segmentor'
        if model_path is not None:
            self.models_dict[model_to_test].load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        total_metrics = {}
        idx = 0
        key_in_path = '' if model_to_test == 'model' else f'_{model_to_test}'
        loader_name = loader_name + key_in_path
        if not osp.exists(osp.join(self.pred_path, loader_name)):
            os.makedirs(osp.join(self.pred_path, loader_name))
        logger_new = TxtLogger(osp.join(self.pred_path, loader_name, f'results.txt'), verbose=True)
        with tqdm(loader) as t:
            for batch in t:
                image = batch['image']
                label = batch['label']
                gnn_dic = batch['gnn']
                gnn2_dic = batch.get('gnn2', None)
                gnn_dic_lists = []
                gnn_dic_lists.append([gnn_dic, gnn2_dic] if gnn2_dic is not None else [gnn_dic])
                idx += label.shape[0]
                image = self._normalize(image)
                image = image.to(device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    pred, _ = self.module_for_segmentor_test(image, gnn_dic_lists[0])
                pred = pred.cpu().to(dtype=torch.float32)
                pred = threshold_torch(pred, thres)
                pred = self._postprocess(pred)
                label = label.cpu().to(dtype=torch.float32)
                for i in range(pred.shape[0]):
                    pred_i = pred[i:i + 1]
                    label_i = label[i:i + 1]
                    if pred_i.shape != label_i.shape:
                        raise ValueError(f'Prediction shape {pred_i.shape} does not match label shape {label_i.shape}')
                    dice = dice_torch(pred_i, label_i)
                    metrics = calc_metrics_torch(pred_i, label_i)
                    logger_new.write(f"Dice for {batch['name'][i]}: {dice:.4f}. Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    total_metrics = dict_element_lst_append(total_metrics, metrics)
                pred = pred.squeeze(0).squeeze(0).numpy()
                self._do_save_image(batch, pred, loader_name, postfix='_pred')
        total_metrics_mean = {}
        total_metrics_std = {}
        for key, value in total_metrics.items():
            if isinstance(value, list):
                total_metrics_mean[key] = np.mean(value)
                total_metrics_std[key] = np.std(value)
            else:
                total_metrics_mean[key] = value
                total_metrics_std[key] = 0.0
        test_str = f'Test result of Segmentor: '
        for key, value in total_metrics_mean.items():
            test_str += f'{key}: {value:.4f} +- {total_metrics_std[key]:.4f}, '
        logger_new.write(test_str)
        self.logger.write(test_str)
        return total_metrics_mean

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml')
    return args.parse_args()

def choose_dataset(full_config):
    dataset = full_config.get('dataset', 'DIAS')
    data_path_base = full_config.get('data_root', f'AIF-SFDA-main/datasets/{dataset}')
    train_split, test_split = ('imagesTr', 'imagesTs') if dataset == 'FIVES' else ('train', 'val')
    return (data_path_base, osp.join(data_path_base, train_split), osp.join(data_path_base, test_split),
            {'name': 'png_norm'}, osp.join(data_path_base, 'frangi_train'))


def main():
    args = parse_args()
    full_config = load_yaml(args.config)
    configure_runtime(full_config)
    set_all_seeds(full_config.get('seed', 42))
    validate_inputs(full_config)
    device = torch.device(full_config.get('device', 'cuda:0'))
    is_train_refiner = full_config.get('train_refiner', False)
    patch_size_description = full_config['trainer_configs'].get('patch_size_description')
    net_norm_method = full_config.get('net_norm_method', 'batch')
    if net_norm_method == 'instance':
        set_eval = True
    elif net_norm_method == 'batch':
        set_eval = False
    else:
        raise ValueError(f'Invalid norm layer {net_norm_method}')
    print(f'Using norm layer {net_norm_method}')
    print(f"Using custom patch size {full_config['trainer_configs']['patch_size']}")
    data_path_base, data_path_train, data_path_test, norm_method, frangi_path = choose_dataset(full_config)
    frangi_path_test = osp.join(data_path_base, 'frangi_val')
    if not full_config.get('graph_version'):
        graph_version = 2
        print(f'graph_version not found in config, using default {graph_version}')
    else:
        graph_version = full_config['graph_version']
    if graph_version >= 10000:
        full_config['trainer_configs']['gnn']['use_slic'] = True
    if not full_config.get('highres_graph_sidelen'):
        highres_graph_sidelen = 16
        print(f'highres_graph_sidelen not found in config, using default {highres_graph_sidelen}')
    else:
        highres_graph_sidelen = full_config['highres_graph_sidelen']
    graph_masked_str = '_masked' if full_config.get('graph_masked', False) else ''
    graph_psuncert_str = '_psuncert' if full_config.get('graph_psuncert', False) else ''
    graph_trained_num_str = f"_{full_config['graph_trained_num']}" if full_config.get('graph_trained_num') is not None else ''
    print('Graph masked:', full_config.get('graph_masked', False))
    if not full_config.get('lowres_graph_sidelen'):
        lowres_graph_sidelen = 32
        print(f'lowres_graph_sidelen not found in config, using default {lowres_graph_sidelen}')
    else:
        lowres_graph_sidelen = full_config['lowres_graph_sidelen']
    other_aux = [{'name': 'gnn', 'aux_path': data_path_base + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_train', 'type': 'pkl'}]
    other_aux_test = [{'name': 'gnn', 'aux_path': data_path_base + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_val', 'type': 'pkl'}]
    if full_config['trainer_configs']['gnn'].get('scales', 1) >= 2:
        other_aux.append({'name': 'gnn2', 'aux_path': data_path_base + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_train', 'type': 'pkl'})
        other_aux_test.append({'name': 'gnn2', 'aux_path': data_path_base + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_val', 'type': 'pkl'})
    wanted_size = tuple(full_config.get('image_size', [1024, 1024]))
    train_dataset_snap_inv = Dataset2d(data_path_train, mode='train', test_fd_str='TrMod', allow_no_label=True, only_no_label=True, last_stage_mask_path={'frangi': frangi_path}, other_aux=other_aux, load_mask_pseudo=data_path_base + '/PLPB_mask_train', norm_method=norm_method, wanted_size=wanted_size, augs=full_config['trainer_configs'].get('augs', None))
    test_dataset_snap_inv = Dataset2d(data_path_test, mode='test', test_fd_str='TrMod', norm_method=norm_method, wanted_size=wanted_size, last_stage_mask_path={'frangi': frangi_path_test}, other_aux=other_aux_test)
    train_loader_snap_inv = DataLoader(train_dataset_snap_inv, batch_size=1, shuffle=True, num_workers=full_config.get('num_workers', 4), pin_memory=True, collate_fn=custom_collate_fn)
    test_loader_snap_inv = DataLoader(test_dataset_snap_inv, batch_size=1, shuffle=False, num_workers=full_config.get('num_workers', 4), pin_memory=True, collate_fn=custom_collate_fn)
    run_name = full_config.get('run_name', 'vesselgda')
    if not is_train_refiner:
        base_folder = osp.join('results', run_name)
    else:
        conv_layers_trained = full_config['trainer_configs']['refiner']['conv_layers']
        feature_map_trained = full_config['trainer_configs']['refiner']['feature_map_from']
        dataset = full_config['dataset']
        print('Training refiner {} {} {} using {}...'.format(conv_layers_trained, feature_map_trained, patch_size_description, dataset))
        base_folder = 'results/artery_seg_exp_refiner_2d_{}_{}{}_{}'.format(conv_layers_trained, feature_map_trained, '_' + patch_size_description if patch_size_description is not None else '', dataset)
    base_folder = full_config.get('output_dir', base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    write_yaml(join(base_folder, 'config.yaml'), full_config)
    trainer_configs = full_config['trainer_configs']
    if trainer_configs['refine_and_pseudo'] and trainer_configs['reweight_pseudo']:
        raise ValueError('Cannot reweight pseudo label when refine_and_pseudo is True')
    training_configs = full_config['training_configs']
    trainer_configs.update(training_configs)
    model = make_net_2d(net_conf=patch_size_description if patch_size_description is not None else 'FIVES', norm_layer=net_norm_method, do_fix_bn=True, image_size=wanted_size)
    if training_configs['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], betas=training_configs['optimizer_params']['betas'] if training_configs.get('optimizer_params') else (0.9, 0.999))
    elif training_configs['optimizer_type'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    elif training_configs['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], momentum=0.9)
    elif training_configs['optimizer_type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    model = model.to(device)
    criterion = None
    trainer = TrainerGNN(model, optimizer, criterion, device, train_loader_snap_inv, test_loader_snap_inv, base_folder=base_folder, set_to_eval_when_testing=set_eval, **trainer_configs)
    if trainer_configs.get('trained_path'):
        trainer.load_models(trainer_configs['trained_path'])
    if trainer_configs.get('pretrain_gnn_path'):
        pretrain_refiner_path = trainer_configs['pretrain_gnn_path'].replace('gnn', 'refiner')
        trainer.load_gnn(trainer_configs['pretrain_gnn_path'], pretrain_refiner_path)
    if is_train_refiner:
        trainer.train(val_interval=None, trained_network='refiner')
    else:
        trainer.train()
        trainer.test(test_loader_snap_inv, 'SnapInv')
        if full_config['trainer_configs'].get('ema'):
            trainer.test(test_loader_snap_inv, 'SnapInv', model_to_test='original')
        if full_config['trainer_configs'].get('segmentor') and full_config['trainer_configs']['segmentor'].get('use'):
            trainer.test_segmentor(test_loader_snap_inv, loader_name='SnapInv', model_path=None)
if __name__ == '__main__':
    main()
