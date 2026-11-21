from submission_config import configure_runtime
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchio as tio
from os.path import join
import numpy as np
import random
import argparse
from sklearn.metrics import roc_auc_score
from dataset import CustomDataset
from custom_loss import WeightedDiceLoss, BinaryFocalLoss, BCELossPosWeight, DiceLoss1d, masked_cross_entropy_dice, graph_smoothness_loss, SoftClDice, SoftClDiceWeighted, SoftcbDiceLoss
from trainer import Trainer, WrappedModelBase
from utils_preprocess import threshold_torch
from utils import safe_mean_torch, dict_element_lst_append, safe_index, dict_stack, dict_to_device, dict_element_add, dict_element_divide, dic_to_string
from baseline_dual_normalization.make_nets import make_unet
from nnunet_network_build.resunet import ResidualEncoderUNetLeak
from nnunetv2.paths import nnUNet_results
from baseline_plpb import get_mask_from_uncertainty
from baseline_tt_sfuda import cross_entropy_dice, softmax_entropy_loss, entropy_loss_no_sigmoid, build_pseduo_augmentation, uncert_voting, build_pseduo_augmentation_plpb, flip_back_plpb, uncert_voting_use_pseudo
from utils import write_nii, load_yaml, write_yaml, check_double_sigmoid
from utils_preprocess import flip_3D_torch
from utils_torch_versions_of_funcs import distance_transform_edt_torch
from seg_networks.graph_unet import TwoScaleGATInterp
from seg_networks.whitening import RandomContrastiveLoss, RandomContrastiveLossMoco, AugRandomContrastiveLossMoco
from utils import EMA_update_teacher_model
from augmentations import GIN, Bezier, RandAug, ShuffleRemap

def get_indices_of_pairs_3d(radius, size):
    """
    Get indices of pairs for a 3D grid within a specified radius.

    Parameters:
    - radius: The search radius.
    - size: A tuple representing the size of the 3D array (depth, height, width).

    Returns:
    - indices_from: A flattened array of the "from" indices.
    - concat_indices_to: A flattened array of the "to" indices, concatenated for all directions.
    """
    search_dist = []
    for z in range(-radius + 1, radius):
        for y in range(-radius + 1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y + z * z < radius * radius:
                    search_dist.append((z, y, x))
    radius_floor = radius - 1
    full_indices = np.reshape(np.arange(0, np.prod(size), dtype=np.int64), size)
    cropped_depth = size[0] - radius_floor
    cropped_height = size[1] - radius_floor
    cropped_width = size[2] - radius_floor
    indices_from = np.reshape(full_indices[radius_floor:-radius_floor, radius_floor:-radius_floor, radius_floor:-radius_floor], [-1])
    indices_to_list = []
    for dz, dy, dx in search_dist:
        indices_to = full_indices[dz + radius_floor:dz + cropped_depth, dy + radius_floor:dy + cropped_height, radius_floor + dx:dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])
        indices_to_list.append(indices_to)
    concat_indices_to = np.concatenate(indices_to_list, axis=0)
    return (indices_from, concat_indices_to)

def get_local_maxima_grid(points, dist_transform, d, D, H, W):
    """Search a square or cubic neighborhood around each node for the maximum distance-transform value."""
    N = points.shape[0]
    side_length = 2 * d + 1
    grid_size = side_length ** 3
    z_coords, y_coords, x_coords = torch.meshgrid(torch.arange(-d, d + 1, device=points.device, dtype=torch.float32), torch.arange(-d, d + 1, device=points.device, dtype=torch.float32), torch.arange(-d, d + 1, device=points.device, dtype=torch.float32), indexing='ij')
    distances = torch.sqrt(x_coords ** 2 + y_coords ** 2 + z_coords ** 2)
    mask = distances <= d
    x_grid = x_coords.reshape(-1)
    y_grid = y_coords.reshape(-1)
    z_grid = z_coords.reshape(-1)
    mask_flat = mask.reshape(-1)
    x_all = points[:, 0:1] + x_grid.unsqueeze(0)
    y_all = points[:, 1:2] + y_grid.unsqueeze(0)
    z_all = points[:, 2:3] + z_grid.unsqueeze(0)
    grid = torch.stack([x_all, y_all, z_all], dim=-1)
    dist_transform_batch = dist_transform.unsqueeze(0).unsqueeze(0)
    grid_batch = grid.unsqueeze(0).to(dtype=torch.float32)
    B, N, grid_size, C = grid_batch.shape
    grid_batch = grid_batch.view(B, N * grid_size, C)
    sampled_values = sample_in_feature_map_3d_torch(dist_transform_batch, grid_batch, mode='nearest')
    sampled_values = sampled_values.view(B, N, grid_size)
    sampled_values = sampled_values.squeeze(0)
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
        self.ind_from, self.ind_to = get_indices_of_pairs_3d(radius=radius, size=feature_size)
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03, betas=(0.9, 0.99))

    def _make_conv_layers(self, in_channels, out_channels, conv_layers):
        assert all((c in ['1', '3'] for c in conv_layers))
        layers = []
        for i in range(len(conv_layers)):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=int(conv_layers[i]), padding=1 if int(conv_layers[i]) == 3 else 0))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=int(conv_layers[i]), padding=1 if int(conv_layers[i]) == 3 else 0))
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
        if x.size(2) == self.feature_size[0] and x.size(3) == self.feature_size[1] and (x.size(4) == self.feature_size[2]):
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

class ResBlock3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        identity = self.conv3(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        return x

class SegmentorFromRefinerMap3(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels_345, softmax=False):
        super(SegmentorFromRefinerMap3, self).__init__()
        assert len(hidden_channels_345) == 3
        self.conv1 = ResBlock3d(in_channels, hidden_channels_345[0])
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = ResBlock3d(hidden_channels_345[0] + hidden_channels_345[1], hidden_channels_345[1])
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = ResBlock3d(hidden_channels_345[1] + hidden_channels_345[2], hidden_channels_345[2])
        self.outconv = nn.Conv3d(hidden_channels_345[2], out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) if softmax else nn.Identity()

    def forward(self, x, outs):
        x = self.conv1(x)
        x = self.up1(x)
        x = torch.cat([x, outs[0]], dim=1)
        x = self.conv2(x)
        x = self.up2(x)
        x = torch.cat([x, outs[1]], dim=1)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.outconv(x)
        x = self.softmax(x)
        return x

class SegmentorFromRefinerMap4(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels_45, softmax=False):
        super(SegmentorFromRefinerMap4, self).__init__()
        assert len(hidden_channels_45) == 2
        self.conv1 = ResBlock3d(in_channels, hidden_channels_45[0])
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = ResBlock3d(hidden_channels_45[0] + hidden_channels_45[1], hidden_channels_45[1])
        self.outconv = nn.Conv3d(hidden_channels_45[1], out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) if softmax else nn.Identity()

    def forward(self, x, outs):
        x = self.conv1(x)
        x = self.up1(x)
        x = torch.cat([x, outs[0]], dim=1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outconv(x)
        x = self.softmax(x)
        return x

def sample_in_feature_map_3d_torch(x, coords, coords_max=None, mode='bilinear'):
    """
    # Checked with example: 
    x = torch.arange(64).reshape([1, 1, 4, 4, 4]).to(dtype=torch.float32)
    coords = torch.tensor([[[0.5, 0.5, 2.5], [1, 3, 1], [3, 1, 1]]])
    coords_max = None ( = 4)
    ans: 12.5, 29, 53

    :param x: [B, C, D, H, W]
    :param coords: [B, N, 3]
    :param coords_max: int or 3-tuple, max value of coords
    :return: feature maps at the N points, [B, N, C]
    """
    _, C, D, H, W = x.shape
    if coords_max is None:
        coords_max = (D, H, W)
    elif isinstance(coords_max, int):
        coords_max = (coords_max, coords_max, coords_max)
    coords_max_arr = torch.tensor(coords_max, dtype=torch.float32, device=x.device).reshape([1, 1, 3])
    coords = (coords + 0.5) / coords_max_arr * 2 - 1
    coords = torch.flip(coords, dims=[2])
    B, N, _ = coords.shape
    coords = coords.reshape([B, N, 1, 1, -1])
    output_features = F.grid_sample(x, coords, mode=mode)
    output_features = output_features.reshape([B, C, N]).permute([0, 2, 1])
    return output_features

def is_coords_in_location(coords, location):
    if coords[0] >= location[0] and coords[1] >= location[1] and (coords[2] >= location[2]) and (coords[0] <= location[3]) and (coords[1] <= location[4]) and (coords[2] <= location[5]):
        return True
    return False

def get_points_and_edges_within_patch(gnn_info_dict, location, device, logger):
    coords = gnn_info_dict['coords'][0]
    gt = gnn_info_dict['gt'][0]
    uncertainty = gnn_info_dict['uncertainty'][0]
    new_info_dict = {}
    new_idxs = []
    for idx in range(len(coords)):
        if is_coords_in_location(coords[idx], location):
            dic0 = {'coords': coords[idx] - location[:3], 'gt': gt[idx], 'uncertainty': uncertainty[idx]}
            new_info_dict = dict_element_lst_append(new_info_dict, dic0)
            new_idxs.append(idx)
    edges = gnn_info_dict['edges'][0]
    edge_weights = gnn_info_dict['edge_weights'][0]
    for idx in range(len(edges)):
        if edge_weights[idx] < 0.05:
            continue
        idx1 = safe_index(new_idxs, edges[idx][0].item())
        idx2 = safe_index(new_idxs, edges[idx][1].item())
        if idx1 != -1 and idx2 != -1:
            dic0 = {'edges': torch.tensor([idx1, idx2]), 'edge_weights': edge_weights[idx]}
            new_info_dict = dict_element_lst_append(new_info_dict, dic0)
    new_info_dict = dict_stack(new_info_dict)
    if new_info_dict.get('edges', None) is None:
        logger.write('Warning: No edges in the patch {}, add a dummy edge.'.format(location))
        new_info_dict['edges'] = torch.tensor([[0, 0]]).to(dtype=torch.long)
        new_info_dict['edge_weights'] = torch.tensor([1.0]).to(dtype=torch.float32)
    new_info_dict = dict_to_device(new_info_dict, device)
    new_info_dict['edges'] = new_info_dict.get('edges', torch.tensor([[new_idxs[0], new_idxs[0]]])).to(dtype=torch.long)
    return new_info_dict

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
        final_pred = torch.softmax(final_pred, dim=1)
        final_pred = final_pred[:, 1:2]
        return threshold_torch(final_pred, 0.5)

class ModuleForSegmentorTest(nn.Module):

    def __init__(self, model, refiner, segmentor, feature_map_from):
        super(ModuleForSegmentorTest, self).__init__()
        self.model = model
        self.refiner = refiner
        self.segmentor = segmentor
        self.feature_map_from = feature_map_from

    def forward(self, x):
        pred, leak_dic = self.model(x)
        feature_map = leak_dic[self.feature_map_from]
        refined_feature = self.refiner(feature_map, torch.zeros_like(pred[:, 1:2]), torch.zeros_like(pred[:, 1:2]), forward_only=True)
        if self.feature_map_from == 'map_3':
            refined_pseudo = self.segmentor(refined_feature, [leak_dic['map_4'], leak_dic['map_5']])
        elif self.feature_map_from == 'map_4':
            refined_pseudo = self.segmentor(refined_feature, [leak_dic['map_5']])
        return (refined_pseudo, None)

def make_net(net_conf='small', track=False):
    folder = os.environ.get('VESSELGDA_SOURCE_MODEL_DIR', 'checkpoints/tof_nnunet')
    checkpoint_path = os.environ.get('VESSELGDA_SOURCE_CHECKPOINT', osp.join(folder, 'fold_0', 'checkpoint_best.pth'))
    unet = make_unet(folder, checkpoint_path, network_class_type=ResidualEncoderUNetLeak, out_channels=2)
    unet.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['network_weights'])
    return unet


class TrainerGNN(Trainer):

    def __init__(self, model, optimizer, criterion, device, train_loader, test_loader, **kwargs):
        super().__init__(model, optimizer, criterion, device, train_loader, test_loader, **kwargs)
        base_folder = kwargs['base_folder']
        if kwargs.get('patch_size_description', 'large') == 'large':
            in_channels_dic = {'map_5': (32, 32), 'map_4': (64, 64), 'map_3': (128, 64), 'map_2': (256, 128), 'map_1': (320, 128), 'map_0': (320, 128)}
        elif kwargs.get('patch_size_description', 'large') == 'small':
            in_channels_dic = {'map_4': (32, 32), 'map_3': (64, 64), 'map_2': (128, 128), 'map_1': (256, 128), 'map_0': (320, 128)}
        else:
            raise ValueError('patch_size_description should be large or small')
        self.in_channels_dic = in_channels_dic
        if kwargs['gnn'].get('type') != 'GATInterp':
            raise NotImplementedError('This release includes the two-scale GATInterp teacher used by the method.')
        assert kwargs['gnn'].get('scales', 1) == 2
        hidden_ch = kwargs['gnn'].get('hidden_ch', in_channels_dic[kwargs['gnn']['map_from']][1])
        self.gnn = TwoScaleGATInterp(in_channels=in_channels_dic[kwargs['gnn']['map_from']][0], hidden_channels=hidden_ch, out_channels=1, num_layers=2, v2=kwargs['gnn'].get('v2', False), heads=kwargs['gnn']['heads'], feature_down=kwargs['gnn']['feature_down'], feature_up=kwargs['gnn']['feature_up'], topk=kwargs['gnn'].get('topk', None), density_correction=kwargs['gnn'].get('density_correction', 'uniform'), learn_kernel=kwargs['gnn'].get('learn_kernel', None))
        self.gnn = self.gnn.to(device=self.device)
        self.optimizer_gnn = torch.optim.Adam(self.gnn.parameters(), lr=kwargs['gnn']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=kwargs['weight_decay'])
        if kwargs['discriminator']['use']:
            raise NotImplementedError('This optional baseline experiment is outside this release.')
        if kwargs.get('refiner') is not None and kwargs.get('refiner').get('use'):
            if kwargs.get('gnn') is not None and kwargs.get('gnn').get('map_from') is not None:
                assert kwargs['refiner']['feature_map_from'] == kwargs['gnn']['map_from']
            if kwargs.get('patch_size_description', 'large') == 'large':
                refiner_kwargs = {'map_0': {'in_channels': 320, 'out_channels': 128, 'feature_size': (4, 10, 8), 'radius': 2}, 'map_1': {'in_channels': 320, 'out_channels': 128, 'feature_size': (4, 20, 16), 'radius': 2}, 'map_2': {'in_channels': 256, 'out_channels': 128, 'feature_size': (8, 40, 32), 'radius': 3}, 'map_3': {'in_channels': 128, 'out_channels': 128, 'feature_size': (16, 80, 64), 'radius': 3}, 'map_4': {'in_channels': 64, 'out_channels': 64, 'feature_size': (32, 160, 128), 'radius': 3}, 'map_5': {'in_channels': 32, 'out_channels': 64, 'feature_size': (64, 320, 256), 'radius': 4}}
            elif kwargs.get('patch_size_description', 'large') == 'small':
                refiner_kwargs = {'map_0': {'in_channels': 320, 'out_channels': 128, 'feature_size': (5, 14, 12), 'radius': 2}, 'map_1': {'in_channels': 256, 'out_channels': 128, 'feature_size': (10, 28, 24), 'radius': 2}, 'map_2': {'in_channels': 128, 'out_channels': 128, 'feature_size': (20, 56, 48), 'radius': 3}, 'map_3': {'in_channels': 64, 'out_channels': 64, 'feature_size': (40, 112, 96), 'radius': 3}, 'map_4': {'in_channels': 32, 'out_channels': 64, 'feature_size': (40, 224, 192), 'radius': 3}}
            else:
                raise ValueError('patch_size_description should be large or small')
            self.refiner = RefineFeature(**refiner_kwargs[kwargs['refiner']['feature_map_from']], conv_layers=kwargs['refiner'].get('conv_layers', '1'))
            self.refiner = self.refiner.to(device=self.device)
            self.optimizer_refiner = torch.optim.Adam(self.refiner.parameters(), lr=kwargs['refiner']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=kwargs['weight_decay'])
            if kwargs['refiner'].get('init_dict_path') is not None:
                self.refiner.load_state_dict(torch.load(kwargs['refiner']['init_dict_path'], map_location=torch.device('cpu')))
            if kwargs['refiner'].get('segmentor') and kwargs['refiner'].get('segmentor').get('use'):
                if kwargs['refiner']['feature_map_from'] == 'map_3':
                    self.segmentor = SegmentorFromRefinerMap3(in_channels=refiner_kwargs['map_3']['out_channels'], out_channels=2, hidden_channels_345=[128, 64, 32])
                elif kwargs['refiner']['feature_map_from'] == 'map_4':
                    self.segmentor = SegmentorFromRefinerMap4(in_channels=refiner_kwargs['map_4']['out_channels'], out_channels=2, hidden_channels_45=[64, 32])
                else:
                    raise NotImplementedError('Only map_3 and map_4 are supported')
                self.segmentor = self.segmentor.to(device=self.device)
                self.optimizer_segmentor = torch.optim.Adam(self.segmentor.parameters(), lr=kwargs['refiner']['lr'], betas=(0.9, 0.99), eps=1e-08, weight_decay=kwargs['weight_decay'])
                self.module_for_segmentor_test = ModuleForSegmentorTest(self.model, self.refiner, self.segmentor, kwargs['refiner']['feature_map_from'])
                self.models_dict['segmentor'] = self.module_for_segmentor_test
                self.model_path['segmentor'] = osp.join(base_folder, 'state_dict_segemtor_model_init.pt')
        if kwargs.get('training_base') == 'aif_sfda':
            raise NotImplementedError('This optional baseline experiment is outside this release.')

    def train_epoch(self, epoch, trained_network='main'):
        self.model.train()
        total_losses = {}
        total_dice = 0
        device = self.device
        idx = 0
        if self.kwargs.get('ema') and self.kwargs['ema_conf'].get('begin_epoch') is not None and (epoch == self.kwargs['ema_conf']['begin_epoch']):
            self.logger.write('EMA model initialized at epoch {}'.format(epoch))
            self.orig_model.load_state_dict(self.model.state_dict())
            self.teacher_model.load_state_dict(self.model.state_dict())
        with tqdm(self.train_loader) as t:
            for batch in t:
                idx += 1
                image = batch['image']
                label = batch['area_interest']
                distmap = batch['frangi']
                gnn_dic = batch['gnn']
                gnn2_dic = batch.get('gnn2', None)
                uncertainty_mask = batch['uncertainty_mask'][0][0]
                pseudo_label = batch['pseudo_label'][0][0]
                distmap_1 = torch.zeros_like(distmap)
                distmap_1[:, :, 1:] = distmap[:, :, :-1]
                distmap = distmap_1
                image = self._normalize(image)
                subject = tio.Subject(image=tio.ScalarImage(tensor=image.squeeze(0)), label=tio.LabelMap(tensor=label.squeeze(0)), distmap=tio.ScalarImage(tensor=distmap.squeeze(0)), uncertainty_mask=tio.LabelMap(tensor=uncertainty_mask.squeeze(0)), pseudo_label=tio.LabelMap(tensor=pseudo_label.squeeze(0)))
                samples = self.sampler(subject, self.patches_in_batch)
                image_lst = []
                label_lst = []
                distmap_lst = []
                gnn_dic_lists = []
                uncertainty_mask_lst = []
                pseudo_label_lst = []
                for sample in samples:
                    image_lst.append(sample['image']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32))
                    label_lst.append(sample['label']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32))
                    distmap_lst.append(sample['distmap']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32))
                    uncertainty_mask_lst.append(sample['uncertainty_mask']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32))
                    pseudo_label_lst.append(sample['pseudo_label']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32))
                    gnn_dic_patch = get_points_and_edges_within_patch(gnn_dic, sample['location'], device, self.logger)
                    gnn_dic_lst = []
                    gnn_dic_lst.append(gnn_dic_patch)
                    if gnn2_dic is not None:
                        gnn2_dic_patch = get_points_and_edges_within_patch(gnn2_dic, sample['location'], device, self.logger)
                        gnn_dic_lst.append(gnn2_dic_patch)
                    gnn_dic_lists.append(gnn_dic_lst)
                self.optimizer.zero_grad()
                if self.kwargs['discriminator']['use']:
                    discriminator_configs = {'net': self.discriminator, 'weight': self.kwargs['discriminator']['weight']}
                else:
                    discriminator_configs = {'net': None, 'weight': 0.5}
                if self.kwargs.get('training_base') in [None, 'plpb', 'tt_sfuda'] or trained_network == 'refiner':
                    for i in range(len(image_lst)):
                        if trained_network == 'main':
                            frangi = distmap_lst[i]
                            if self.kwargs.get('training_base') in [None, 'plpb']:
                                losses_dic = self._do_train(self.model, epoch, image_lst[i], uncertainty_mask_lst[i], pseudo_label_lst[i], self.gnn, gnn_dic_lists[i], mode='pred', lr=self.kwargs['lr'], weight_decay=self.kwargs['weight_decay'], discriminator=discriminator_configs, gnn=self.kwargs['gnn'], cpr_refine={'map': distmap_lst[i] if self.kwargs['cpr_refine']['use'] else None, 'entropy_thres': self.kwargs['cpr_refine']['thres']}, frangi=frangi)
                            elif self.kwargs['training_base'] == 'tt_sfuda':
                                losses_dic = self._do_train_tt_sfuda(self.model, self.orig_model, self.gnn, image_lst[i], gnn_dic_lists[i], epoch, frangi=frangi)
                        elif trained_network == 'refiner':
                            losses_dic = _do_train_refine_feature(self.model, self.refiner, self.kwargs['refiner']['feature_map_from'], image_lst[i], uncertainty_mask_lst[i], pseudo_label_lst[i], lr=self.kwargs['refiner']['lr'], weight_decay=self.kwargs['weight_decay'])
                            torch.save(self.refiner.state_dict(), join(self.base_folder, f'refiner_{epoch}.pth'))
                        total_losses = dict_element_add(total_losses, losses_dic)
                elif self.kwargs['training_base'] == 'aif_sfda':
                    raise NotImplementedError('This optional baseline experiment is outside this release.')
                t.set_postfix(dict_element_divide(total_losses, idx))
        if self.kwargs.get('ema'):
            if self.kwargs['ema_conf'].get('begin_epoch') is None or epoch >= self.kwargs['ema_conf'].get('begin_epoch', 0):
                new_dict = EMA_update_teacher_model(self.model, self.teacher_model, self.kwargs['ema'])
                self.teacher_model.load_state_dict(new_dict)
                if self.kwargs.get('ema_conf', True) and self.kwargs['ema_conf'].get('update_per', 1) is not None and ((epoch + 1) % self.kwargs['ema_conf'].get('update_per', 1) == 0):
                    new_dict_orig = EMA_update_teacher_model(self.model, self.orig_model, self.kwargs['ema'])
                    self.orig_model.load_state_dict(new_dict_orig)
        train_str = f'Epoch {epoch}\n' + dic_to_string(dict_element_divide(total_losses, idx))
        self.logger.write(train_str)

    def _do_train(self, model, epoch, x, uncertainty_mask, pseudo_label, gnn_model, gnn_info, lr=5e-06, weight_decay=1e-05, mode='pred', refined_and_pseudo=False, reweight_pseudo=False, frangi=None, discriminator={'net': None, 'weight': 0.5}, gnn={'weight': 1.0, 'map_from': 'map_5', 'lr': 1e-05}, cpr_refine={'map': None, 'weight': 1.0, 'entropy_thres': 0.1, 'refine_feature_map': False}):
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
        if True:
            mask, pseudo_label_1, weighted_ent = self._calc_pseudo_mask_online(x, self.orig_model)
        else:
            mask = uncertainty_mask
            pseudo_label_1 = pseudo_label
            weighted_ent = torch.clone(mask)
        mask = mask.to(device=device, dtype=torch.float32)
        pseudo_label_1 = pseudo_label_1.to(device=device, dtype=torch.float32)
        x_ori = x.to(device=device, dtype=torch.float32)
        pred, leak_dic = model(x_ori)
        pred = torch.softmax(pred, dim=1)
        sh = pred.shape
        write_nii((pred[:, 1:2] > 0.5).float(), 'pred.nii.gz')
        write_nii(pseudo_label_1[:, 1:2], 'pseudo_label.nii.gz')
        loss_dic = {}
        if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
            is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
            frangi = frangi.to(device)[0]
            mask_output_this = ((mask[:, 1] > 0.5) | (pseudo_label_1[:, 1] > 0.5)).to(torch.float32)
            pseudo_label_1 = self._mix_frangi_and_pseudo(frangi, pseudo_label_1[:, 1], mask_output_this, hard_frangi=is_hard_frangi)
            pseudo_label_1 = pseudo_label_1.unsqueeze(1)
            pseudo_label_1 = torch.cat([1 - pseudo_label_1, pseudo_label_1], dim=1)
        if self.kwargs.get('map_contrastive_loss') and self.kwargs['map_contrastive_loss']['use']:
            if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                aug_pred_conf = self.kwargs['map_contrastive_loss'].get('aug_pred_type', {'type': 'follow'})
                aug_pred_type = aug_pred_conf.get('type', 'follow')
                if aug_pred_type == 'follow':
                    r = random.randint(1, 7)
                    aug_input_1 = flip_3D_torch(x, r).to(device)
                    pred_1, leak_dic_1 = model(aug_input_1)
                    pred_1 = torch.softmax(pred_1, dim=1)
                    if self.kwargs.get('test_time_aug_type') == 'flip':
                        pred_1 = flip_3D_torch(pred_1, r)
                        for key in leak_dic_1.keys():
                            leak_dic_1[key] = flip_3D_torch(leak_dic_1[key], r)
            map_contrastive_loss_conf = self.kwargs['map_contrastive_loss']
            assert map_contrastive_loss_conf['map_from'] in ['map_5', 'map_4', 'map_3']
            if map_contrastive_loss_conf['map_from'] == 'map_5':
                feature_to_refine = leak_dic['map_5']
                crop_size = (8, 20, 20)
            elif map_contrastive_loss_conf['map_from'] == 'map_4':
                feature_to_refine = leak_dic['map_4']
                crop_size = (4, 10, 10)
            elif map_contrastive_loss_conf['map_from'] == 'map_3':
                feature_to_refine = leak_dic['map_3']
                crop_size = (2, 5, 5)
            if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                assert map_contrastive_loss_conf['kl_type'] in ['moco']
                contrastive_loss_func = AugRandomContrastiveLossMoco(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=map_contrastive_loss_conf.get('uncertain_prob', 0), wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss=map_contrastive_loss_conf.get('aug_l1_loss', {'use': False}))
                contrastive_loss = contrastive_loss_func(feature_to_refine, leak_dic_1[map_contrastive_loss_conf['map_from']], pseudo_label_1[:, 1].unsqueeze(0), mask[:, 1].unsqueeze(0).to(dtype=torch.float32), entropy_map=weighted_ent[:, 1].unsqueeze(0).to(dtype=torch.float32)) * map_contrastive_loss_conf['weight']
            loss_dic['map_contr_loss'] = contrastive_loss.item()
            contrastive_loss.backward(retain_graph=True)
        bce_loss = nn.BCELoss()
        ps_output = pseudo_label_1[:, 1]
        mask_output = mask[:, 1]
        if gnn_conf['weight'] > 0:
            loss_gnn, losses_dic = self._calc_loss_gnn(leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic)
        else:
            loss_gnn = torch.tensor(0, dtype=pred.dtype, device=pred.device)
            losses_dic = {}
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
        idxs = torch.where(edge_weights > thres)[0]
        new_gnn_info = {}
        for key in gnn_info.keys():
            if key in ['edges', 'edge_weights']:
                new_gnn_info[key] = gnn_info[key][idxs]
            else:
                new_gnn_info[key] = gnn_info[key]
        return new_gnn_info

    def _mix_frangi_and_pseudo(self, frangi, pseudo, mask, hard_frangi=False):
        sigmoid_s = self.kwargs['frangi_in_pseudo']['sigmoid_s']
        sigmoid_k = self.kwargs['frangi_in_pseudo']['sigmoid_k']
        if sigmoid_k is None:
            if sigmoid_s > 0:
                sigmoid_k = 2.5 / sigmoid_s
            else:
                sigmoid_k = 1.0
        frangi = 1 / (1 + torch.exp(-(frangi - sigmoid_s) * sigmoid_k))
        if hard_frangi:
            frangi = (frangi > 0.5).to(dtype=frangi.dtype)
        pseudo = pseudo * mask
        frangi = frangi * (1 - mask)
        return frangi + pseudo

    def _mix_frangi_and_pseudo_add_area_interest(self, frangi, pseudo, mask, hard_frangi=False, force_pseudo_true=False):
        sigmoid_s = self.kwargs['frangi_in_pseudo']['sigmoid_s']
        sigmoid_k = self.kwargs['frangi_in_pseudo']['sigmoid_k']
        if sigmoid_k is None:
            if sigmoid_s > 0:
                sigmoid_k = 2.5 / sigmoid_s
            else:
                sigmoid_k = 1.0
        frangi = 1 / (1 + torch.exp(-(frangi - sigmoid_s) * sigmoid_k))
        frangi_bool = frangi > 0.5
        if hard_frangi:
            frangi = (frangi > 0.5).to(dtype=frangi.dtype)
        certain_positive_area = (pseudo > 0.5) & (mask > 0.5)
        mask[frangi_bool & ~certain_positive_area] = 0.0
        if force_pseudo_true:
            pseudo[frangi_bool & (pseudo < 0.5)] = frangi[frangi_bool & (pseudo < 0.5)]
        else:
            pseudo[frangi_bool] = frangi[frangi_bool].to(dtype=pseudo.dtype)
        return (pseudo, mask)

    def _do_train_tt_sfuda(self, model, original_model, gnn_model, x, gnn_info, epoch, frangi=None):
        self.optimizer.zero_grad()
        self.optimizer_gnn.zero_grad()
        device = self.device
        gnn_conf = self.kwargs['gnn']
        if gnn_conf.get('delete_edge_weight_under', 0) > 0:
            for gnn_info_idx in range(len(gnn_info)):
                gnn_info[gnn_info_idx] = self._do_gnn_delete_edge_weight_under(gnn_info[gnn_info_idx], gnn_conf['delete_edge_weight_under'])
        criterion = cross_entropy_dice
        if self.kwargs.get('test_time_aug_type') == 'flip':
            aug_input = build_pseduo_augmentation_plpb(x.cpu().numpy())
        elif self.kwargs.get('test_time_aug_type') in [None, 'orig']:
            aug_input = build_pseduo_augmentation(x.cpu().numpy())
        else:
            raise ValueError('Invalid test_time_aug_type')
        loss_dic = {}
        with torch.no_grad():
            aug_outputs = []
            aug_feature_outputs = []
            for i in range(aug_input.shape[0]):
                aug_input_i = aug_input[i].to(device).unsqueeze(0)
                aug_output_i, leak_dic = original_model(aug_input_i)
                aug_outputs.append(aug_output_i.detach().cpu())
                aug_feature_outputs.append(leak_dic['last_feature_map'].detach())
            aug_output = torch.cat(aug_outputs, dim=0).to(device=self.device)
            aug_feature = torch.cat(aug_feature_outputs, dim=0).to(device=self.device)
            if self.kwargs.get('test_time_aug_type') == 'flip':
                aug_output = flip_back_plpb(aug_output)
                aug_feature = flip_back_plpb(aug_feature)
            voting_method = self.kwargs.get('voting_method', 'tt')
            if voting_method == 'tt':
                ps_output, mask_output, weighted_ent = uncert_voting(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True))
            elif voting_method == 'plpb':
                ps_output, mask_output, weighted_ent = self._calc_pseudo_mask_online_from_pred(aug_output.detach(), aug_feature.detach())
            elif voting_method == 'pseudo':
                ps_output, mask_output, weighted_ent = uncert_voting_use_pseudo(aug_output.detach(), and_fn=self.kwargs.get('tt_pseudo_fn_use', True), thres=self.kwargs.get('pseudo_thres', 0.15))
            mask_output = 1 - mask_output
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
                is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
                frangi = frangi.to(device)[0]
                if self.kwargs['frangi_in_pseudo'].get('no_mod_pos', False):
                    mask_output_this = ((mask_output > 0.5) | (ps_output > 0.5)).to(torch.float32)
                else:
                    mask_output_this = mask_output
                if self.kwargs['frangi_in_pseudo'].get('add_to_area_interest', False) is False:
                    ps_output = self._mix_frangi_and_pseudo(frangi, ps_output, mask_output_this, hard_frangi=is_hard_frangi)
                else:
                    ps_output, mask_output = self._mix_frangi_and_pseudo_add_area_interest(frangi, ps_output, mask_output_this, hard_frangi=is_hard_frangi, force_pseudo_true=self.kwargs['frangi_in_pseudo'].get('force_pseudo_true', False))
        self.optimizer.zero_grad()
        for i in range(aug_input.shape[0]):
            aug_input_i = aug_input[i].to(device).unsqueeze(0)
            output_i, _ = model(aug_input_i)
            if self.kwargs.get('test_time_aug_type') == 'flip':
                ps_output_i = flip_3D_torch(ps_output.clone().unsqueeze(0), i)[0]
                mask_output_i = flip_3D_torch(mask_output.clone().unsqueeze(0), i)[0]
            else:
                ps_output_i = ps_output.clone()
                mask_output_i = mask_output.clone()
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use'] and self.kwargs['frangi_in_pseudo'].get('frangi_loss_use', True):
                smoothl1_no_reduction = nn.SmoothL1Loss(reduction='none')
                seg_loss_1 = masked_cross_entropy_dice(output_i, ps_output_i.to(device=device, dtype=torch.long), mask_output.unsqueeze(1))
                if not is_hard_frangi:
                    seg_loss_2 = smoothl1_no_reduction(torch.softmax(output_i, dim=1)[:, 1] * (1 - mask_output), ps_output_i.to(device) * (1 - mask_output))
                else:
                    smoothing = self.kwargs['frangi_in_pseudo'].get('smooth', 0.0)
                    seg_loss_2 = F.cross_entropy(output_i, ps_output_i.to(device=device, dtype=torch.long), reduction='none', label_smoothing=smoothing) * (1 - mask_output)
                if self.kwargs['frangi_in_pseudo'].get('frangi_loss_weight', 1.0) == -1:
                    self.kwargs['frangi_in_pseudo']['frangi_loss_weight'] = (1 - mask_output).sum().item() / mask_output.numel()
                seg_loss = seg_loss_1 + seg_loss_2.sum() / ((1 - mask_output).sum() + 1e-05) * self.kwargs['frangi_in_pseudo'].get('frangi_loss_weight', 1.0)
            else:
                seg_loss = masked_cross_entropy_dice(output_i, ps_output_i.to(device=device, dtype=torch.long), torch.ones_like(ps_output_i.to(device=device, dtype=torch.long)))
            ent_loss = softmax_entropy_loss(output_i)
            loss = self.kwargs['loss_weights']['seg'] * seg_loss + self.kwargs['loss_weights']['ent'] * ent_loss
            if self.kwargs.get('cldice') and self.kwargs['cldice']['use']:
                certain_positive_region = ps_output_i * mask_output_i
                uncertain_positive_region = ps_output_i * (1 - mask_output_i)
                uncertain_negative_region = (1 - ps_output_i) * (1 - mask_output_i)
                combined_mask_for_cldice = certain_positive_region + self.kwargs['cldice']['beta1'] * uncertain_positive_region + self.kwargs['cldice']['beta2'] * uncertain_negative_region
                if self.kwargs['cldice'].get('type', 'orig') == 'orig':
                    cldice_loss_func = SoftClDice()
                    cldice = cldice_loss_func(combined_mask_for_cldice.to(device=device, dtype=torch.float32).unsqueeze(1), torch.softmax(output_i, dim=1)[:, 1:2])
                elif self.kwargs['cldice']['type'] == 'weighted':
                    cldice_loss_func = SoftClDiceWeighted(weight_type=self.kwargs['cldice'].get('weight_type', 'exp'), power=self.kwargs['cldice'].get('power', 0.1))
                    cldice = cldice_loss_func(combined_mask_for_cldice.to(device=device, dtype=torch.float32).unsqueeze(1), torch.softmax(output_i, dim=1)[:, 1:2])
                elif self.kwargs['cldice']['type'] == 'cbdice':
                    cldice_loss_func = SoftcbDiceLoss(smooth=0.001)
                    cldice = cldice_loss_func(output_i, combined_mask_for_cldice.to(device=device, dtype=torch.float32).unsqueeze(1))
                loss = loss + self.kwargs['cldice']['weight'] * cldice
            loss.backward()
            if self.kwargs.get('step_each_aug', True) and (not self.kwargs.get('step_last_only', False)):
                self.optimizer.step()
        if not self.kwargs.get('step_each_aug', True) and (not self.kwargs.get('step_last_only', False)):
            self.optimizer.step()
        loss_dic['seg_loss'] = seg_loss.item()
        loss_dic['ent_loss'] = ent_loss.item()
        aug_input_gnn = aug_input[0].to(device).unsqueeze(0)
        pred, leak_dic = model(aug_input_gnn)
        pred = torch.softmax(pred, dim=1)
        if self.kwargs.get('map_contrastive_loss') and self.kwargs['map_contrastive_loss']['use'] and (epoch >= self.kwargs['map_contrastive_loss'].get('start_epoch', 0)):
            if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                aug_pred_conf = self.kwargs['map_contrastive_loss'].get('aug_pred_type', {'type': 'follow'})
                aug_pred_type = aug_pred_conf.get('type', 'follow')
                if aug_pred_type == 'follow':
                    r = random.randint(1, aug_input.shape[0] - 1)
                    aug_input_1 = aug_input[r].to(device).unsqueeze(0)
                    pred_1, leak_dic_1 = model(aug_input_1)
                    pred_1 = torch.softmax(pred_1, dim=1)
                    if self.kwargs.get('test_time_aug_type') == 'flip':
                        pred_1 = flip_3D_torch(pred_1, r)
                        for key in leak_dic_1.keys():
                            leak_dic_1[key] = flip_3D_torch(leak_dic_1[key], r)
                elif aug_pred_type in ['GIN', 'bezier', 'randaug', 'shuffle_remap']:
                    if aug_pred_type == 'GIN':
                        aug_class = GIN(1, aug_pred_conf['mid_channels'], aug_pred_conf['num_layers'])
                    elif aug_pred_type == 'bezier':
                        aug_class = Bezier(flip=False, prob=1.0, num_free_coordinates=aug_pred_conf['num_free_coordinates'], max_rnd_val=aug_pred_conf['max_rnd_val'], min_rnd_val=aug_pred_conf['min_rnd_val'])
                    elif aug_pred_type == 'randaug':
                        aug_class = RandAug(aug_pred_conf['n_augs'])
                    elif aug_pred_type == 'shuffle_remap':
                        aug_class = ShuffleRemap(aug_pred_conf['rand_point_lo'], aug_pred_conf['rand_point_hi'])
                    aug_class = aug_class.to(device)
                    aug_input_1 = aug_class(x)
                    pred_1, leak_dic_1 = model(aug_input_1)
                    pred_1 = torch.softmax(pred_1, dim=1)
            map_contrastive_loss_conf = self.kwargs['map_contrastive_loss']
            crop_size = (8, 20, 20)
            contrastive_loss = 0
            if type(map_contrastive_loss_conf['map_from']) == str:
                map_contrastive_loss_conf['map_from'] = [map_contrastive_loss_conf['map_from']]
            for map_from in map_contrastive_loss_conf['map_from']:
                feature_to_refine = leak_dic[map_from]
                if self.kwargs['map_contrastive_loss'].get('aug_pred'):
                    assert map_contrastive_loss_conf['kl_type'] in ['moco']
                    if map_contrastive_loss_conf['kl_type'] == 'moco':
                        if not map_contrastive_loss_conf.get('moco_no_uncertain', False):
                            contrastive_loss_func = AugRandomContrastiveLossMoco(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=map_contrastive_loss_conf.get('uncertain_prob', 0), wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss=map_contrastive_loss_conf.get('aug_l1_loss', {'use': False}), moco_weight=map_contrastive_loss_conf.get('moco_weight', 1.0))
                            contrastive_loss = contrastive_loss + contrastive_loss_func(feature_to_refine, leak_dic_1[map_from], ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), entropy_map=weighted_ent.unsqueeze(0)) * map_contrastive_loss_conf['weight']
                    else:
                        contrastive_loss_func_l1 = AugRandomContrastiveLossMoco(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=map_contrastive_loss_conf.get('uncertain_prob', 0), wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss=map_contrastive_loss_conf.get('aug_l1_loss', {'use': False}), moco_weight=0.0)
                        contrastive_loss_func_moco = AugRandomContrastiveLossMoco(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, uncertain_prob=0, wanted_features=map_contrastive_loss_conf.get('wanted_features', ['mean']), dense_contrastive_loss=map_contrastive_loss_conf.get('dense_contrastive_loss', {'use': False}), aug_l1_loss={'use': False}, moco_weight=map_contrastive_loss_conf.get('moco_weight', 1.0))
                        contrastive_loss = contrastive_loss + contrastive_loss_func_l1(feature_to_refine, leak_dic_1[map_from], ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), entropy_map=weighted_ent.unsqueeze(0)) * map_contrastive_loss_conf['weight']
                        contrastive_loss = contrastive_loss + contrastive_loss_func_moco(feature_to_refine, leak_dic_1[map_from], ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), entropy_map=weighted_ent.unsqueeze(0)) * map_contrastive_loss_conf['weight']
                else:
                    if map_contrastive_loss_conf['kl_type'] in ['kl', 'wass', 'kl_norm', 'cosine']:
                        contrastive_loss_func = RandomContrastiveLoss(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size, loss_type=map_contrastive_loss_conf['kl_type'])
                    elif map_contrastive_loss_conf['kl_type'] in ['moco']:
                        contrastive_loss_func = RandomContrastiveLossMoco(num_fg_blocks=map_contrastive_loss_conf['num_fg_blocks'], num_bg_blocks=map_contrastive_loss_conf['num_bg_blocks'], crop_size=crop_size)
                    contrastive_loss = contrastive_loss + contrastive_loss_func(feature_to_refine, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32)) * map_contrastive_loss_conf['weight']
            loss_dic['map_contr_loss'] = contrastive_loss.item()
            contrastive_loss.backward(retain_graph=True)
        if gnn_conf.get('zero_grad_before_gnn', False):
            if self.kwargs.get('step_last_only', False):
                raise ValueError('step_last_only and zero_grad_before_gnn cannot be both true')
            self.optimizer.zero_grad()
        if gnn_conf['weight'] > 0:
            if gnn_conf.get('pred_edge', False):
                loss_gnn, loss_dic = self._calc_loss_gnn_weak_supervision(leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic, epoch=epoch)
            else:
                aug_pred_lst = []
                if len(gnn_conf.get('aug_pred_types', [])) > 0:
                    for aug_pred_type in gnn_conf['aug_pred_types']:
                        if aug_pred_type == 'follow':
                            r = random.randint(1, aug_input.shape[0] - 1)
                            aug_input_i = aug_input[r].to(device).unsqueeze(0)
                            pred_i, leak_dic_i = model(aug_input_i)
                            pred_i = torch.softmax(pred_i, dim=1)
                            if self.kwargs.get('test_time_aug_type') == 'flip':
                                pred_i = flip_3D_torch(pred_i, r)
                                for key in leak_dic_i.keys():
                                    leak_dic_i[key] = flip_3D_torch(leak_dic_i[key], r)
                        elif aug_pred_type in ['GIN', 'bezier', 'randaug', 'shuffle_remap']:
                            if aug_pred_type == 'GIN':
                                aug_class = GIN(1, gnn_conf['mid_channels'], gnn_conf['num_layers'])
                            elif aug_pred_type == 'bezier':
                                aug_class = Bezier(flip=False, prob=1.0, num_free_coordinates=gnn_conf['num_free_coordinates'], max_rnd_val=gnn_conf['max_rnd_val'], min_rnd_val=gnn_conf['min_rnd_val'])
                            elif aug_pred_type == 'randaug':
                                aug_class = RandAug(gnn_conf['n_augs'])
                            elif aug_pred_type == 'shuffle_remap':
                                aug_class = ShuffleRemap(gnn_conf['rand_point_lo'], gnn_conf['rand_point_hi'])
                            aug_class = aug_class.to(device)
                            aug_input_i = aug_class(x)
                            pred_i, leak_dic_i = model(aug_input_i)
                            aug_pred_lst.append(leak_dic_i[gnn_conf['map_from']])
                        loss_gnn, loss_dic, _ = self._calc_loss_gnn_weak_supervision(leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic, aug_pred_lst=aug_pred_lst, epoch=epoch)
            loss_gnn.backward()
            self.optimizer_gnn.step()
        else:
            loss_gnn = torch.tensor(0, dtype=x.dtype, device=x.device)
        self.optimizer.step()
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            self.optimizer_refiner.step()
            self.optimizer_refiner.zero_grad()
        loss_dic['gnn_loss'] = loss_gnn.item()
        return loss_dic

    def _do_train_aif_sfda(self, model, teacher_model, gnn_model, x, gnn_info, frangi=None):
        raise NotImplementedError('AIF baseline training is outside this release.')

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

    def _calc_loss_gnn_weak_supervision(self, leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic, aug_pred_lst=[], epoch=None, name=None, is_test=False):
        is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
        feature = leak_dic[gnn_conf['map_from']][0].unsqueeze(0)
        smoothl1_no_reduction = nn.SmoothL1Loss(reduction='none')
        smoothl1_loss = nn.SmoothL1Loss()
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            feature = self.refiner(feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
        if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
            feature = nn.Upsample(size=x.shape[2:], mode='trilinear')(feature)
        if gnn_conf.get('scales', 1) == 1:
            assert len(gnn_info) == 1
            gnn_info = gnn_info[0]
            if gnn_conf.get('use_slic', False):
                raise NotImplementedError('SLIC not implemented for 3D yet')
            else:
                interp_mode = gnn_conf.get('interp_mode', 'bilinear')
                feature_interp = sample_in_feature_map_3d_torch(feature, gnn_info['coords'].unsqueeze(0), mode=interp_mode)[0]
                gt_interp = sample_in_feature_map_3d_torch(ps_output.to(dtype=torch.float32).unsqueeze(0), gnn_info['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
                uncertainty_interp = sample_in_feature_map_3d_torch(mask_output.to(dtype=torch.float32).unsqueeze(0), gnn_info['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
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
                        raise NotImplementedError('SLIC not implemented for 3D yet')
                    else:
                        feature_aug = sample_in_feature_map_3d_torch(aug_pred_feature, gnn_info['coords'].unsqueeze(0), mode=interp_mode)[0]
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
                raise NotImplementedError('ELR not implemented yet for GNN')
            elif loss_type == 'focal':
                bceloss0 = BinaryFocalLoss(neg_weight=1, gamma=2, reduction='none')
            elif loss_type == 'dice':
                bceloss0 = DiceLoss1d()
            elif loss_type == 'bce':
                bceloss0 = BCELossPosWeight(pos_weight=1, reduction='none')
            loss_gnn = bceloss0(pred_gnn, gt_interp)
            if gnn_conf.get('gnn_memory_bank') and gnn_conf['gnn_memory_bank']['use'] and (not is_test):
                raise NotImplementedError('GNN memory bank not implemented yet')
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
                    raise NotImplementedError('SLIC not implemented for 3D yet')
                else:
                    pred_interp = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info['coords'].unsqueeze(0), mode='bilinear')[0]
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
                    raise NotImplementedError('SLIC not implemented for 3D yet')
                else:
                    feature_interp = sample_in_feature_map_3d_torch(feature, gnn_info_this_scale['coords'].unsqueeze(0), mode=interp_mode)[0]
                    gt_interp = sample_in_feature_map_3d_torch(ps_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
                    uncertainty_interp = sample_in_feature_map_3d_torch(mask_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
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
                    raise NotImplementedError('Edge augmentation not implemented yet for TwoScaleGATInterp and MGGNN')
                if len(aug_pred_lst) > 0:
                    loss_aug_pred = 0
                    for idx in range(len(aug_pred_lst)):
                        aug_pred_feature = aug_pred_lst[idx]
                        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
                            aug_pred_feature = self.refiner(aug_pred_feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
                        if gnn_conf['map_from'] not in ['map_4', 'last_feature_map']:
                            aug_pred_feature = nn.Upsample(size=x.shape[2:], mode='bilinear')(aug_pred_feature)
                        feature_aug_1 = sample_in_feature_map_3d_torch(aug_pred_feature, gnn_info[0]['coords'].unsqueeze(0), mode=interp_mode)[0]
                        feature_aug_2 = sample_in_feature_map_3d_torch(aug_pred_feature, gnn_info[1]['coords'].unsqueeze(0), mode=interp_mode)[0]
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
                raise NotImplementedError('GNN memory bank not implemented yet')
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
                raise NotImplementedError('Edge augmentation not implemented yet for TwoScaleGATInterp and MGGNN')
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
                    raise NotImplementedError('SLIC not implemented for 3D yet')
                else:
                    pred_interp_1 = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info[0]['coords'].unsqueeze(0), mode='bilinear')[0].squeeze(-1)
                    pred_interp_2 = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info[1]['coords'].unsqueeze(0), mode='bilinear')[0].squeeze(-1)
                if gnn_conf.get('consistency_gnn_detach', False):
                    pred_gnn_1 = pred_gnn_1.detach()
                    pred_gnn_2 = pred_gnn_2.detach()
                if gnn_conf.get('consistency_no_sigmoid', False):
                    pred_interp_1 = torch.logit(pred_interp_1, eps=1e-06)
                    pred_interp_2 = torch.logit(pred_interp_2, eps=1e-06)
                    pred_gnn_1 = torch.logit(pred_gnn_1, eps=1e-06)
                    pred_gnn_2 = torch.logit(pred_gnn_2, eps=1e-06)
                if gnn_conf.get('trunk_connection'):
                    raise NotImplementedError('Trunk connection not implemented yet for 3D')
                elif gnn_conf.get('dt_weight'):
                    assert not gnn_conf.get('trunk_connection')
                    dt_map = distance_transform_edt_torch((ps_output > 0.5).to(dtype=torch.float32).unsqueeze(0))
                    dt_interp = sample_in_feature_map_3d_torch(dt_map, gnn_info[0]['coords'].unsqueeze(0), mode='bilinear')[0].squeeze(-1)
                    dt_local_maxima = get_local_maxima_grid(gnn_info[0]['coords'].to(dt_interp.device), dt_map.squeeze(0).squeeze(0), d=2, D=ps_output.shape[-3], H=ps_output.shape[-2], W=ps_output.shape[-1])
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

    def _calc_loss_gnn(self, leak_dic, gnn_model, gnn_info, gnn_conf, x, pred, ps_output, mask_output, loss_dic):
        is_hard_frangi = self.kwargs['frangi_in_pseudo'].get('hard', False)
        feature = leak_dic[gnn_conf['map_from']][0].unsqueeze(0)
        smoothl1_no_reduction = nn.SmoothL1Loss(reduction='none')
        smoothl1_loss = nn.SmoothL1Loss()
        ce_loss = nn.BCELoss(reduction='none')
        if self.kwargs.get('refiner') is not None and self.kwargs['refiner']['use']:
            feature = self.refiner(feature, ps_output.unsqueeze(0), mask_output.unsqueeze(0).to(dtype=torch.float32), forward_only=True)
        if gnn_conf['map_from'] not in ['map_5', 'last_feature_map']:
            feature = nn.Upsample(size=x.shape[2:], mode='trilinear')(feature)
        if gnn_conf.get('scales', 1) == 1:
            assert len(gnn_info) == 1
            gnn_info = gnn_info[0]
            feature_interp = sample_in_feature_map_3d_torch(feature, gnn_info['coords'].unsqueeze(0), mode='bilinear')[0]
            gt_interp = sample_in_feature_map_3d_torch(ps_output.to(dtype=torch.float32).unsqueeze(0), gnn_info['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
            uncertainty_interp = sample_in_feature_map_3d_torch(mask_output.to(dtype=torch.float32).unsqueeze(0), gnn_info['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
            if gnn_model.__class__.__name__ == 'GAT':
                pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'].T, edge_attr=gnn_info['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None)
            elif gnn_model.__class__.__name__ in ['GraphUNet', 'GATUNet']:
                pred_gnn = gnn_model(feature_interp, edge_index=gnn_info['edges'].T, edge_weight=gnn_info['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None)
            pred_gnn = torch.sigmoid(pred_gnn).squeeze(-1)
            bceloss0 = BinaryFocalLoss(neg_weight=1, gamma=2, reduction='none')
            loss_gnn = bceloss0(pred_gnn, gt_interp)
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
                if is_hard_frangi:
                    loss_gnn_l1 = bceloss0(pred_gnn, gt_interp)
                else:
                    loss_gnn_l1 = smoothl1_no_reduction(pred_gnn, gt_interp)
                loss_gnn_l1 = loss_gnn_l1[uncertainty_interp < 0.5].mean() * gnn_conf['weight']
                loss_gnn = loss_gnn[uncertainty_interp > 0.5].mean() * gnn_conf['weight']
                loss_gnn = loss_gnn + loss_gnn_l1
            elif gnn_conf.get('loss_uncertainty_mask', True):
                loss_gnn = loss_gnn[uncertainty_interp > 0.5].mean() * gnn_conf['weight']
            else:
                loss_gnn = loss_gnn.mean() * gnn_conf['weight']
            if gnn_conf.get('consistency'):
                pred_interp = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info['coords'].unsqueeze(0), mode='bilinear')[0]
                pred_interp = pred_interp.squeeze(-1)
                consistency_loss = smoothl1_loss(pred_interp, pred_gnn) * gnn_conf['consistency']
                consistency_loss.backward(retain_graph=True)
                loss_gnn = loss_gnn + consistency_loss
                loss_dic['consistency_loss'] = consistency_loss.item()
        elif gnn_conf.get('scales', 1) == 2:
            interps = {}
            for gnn_info_this_scale in gnn_info:
                feature_interp = sample_in_feature_map_3d_torch(feature, gnn_info_this_scale['coords'].unsqueeze(0), mode='bilinear')[0]
                gt_interp = sample_in_feature_map_3d_torch(ps_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
                uncertainty_interp = sample_in_feature_map_3d_torch(mask_output.to(dtype=torch.float32).unsqueeze(0), gnn_info_this_scale['coords'].unsqueeze(0), mode='nearest')[0].squeeze(-1)
                interps = dict_element_lst_append(interps, {'feature': feature_interp, 'gt': gt_interp, 'uncertainty': uncertainty_interp})
            if gnn_model.__class__.__name__ == 'TwoScaleGAT':
                pred_gnn_1, pred_gnn_2 = gnn_model(feature1=interps['feature'][0], edge_index1=gnn_info[0]['edges'].T, edge_attr1=gnn_info[0]['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None, feature2=interps['feature'][1], edge_index2=gnn_info[1]['edges'].T, edge_attr2=gnn_info[1]['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None)
            elif gnn_model.__class__.__name__ in ['TwoScaleGATInterp', 'MGGNN']:
                pred_gnn_1, pred_gnn_2 = gnn_model(feature1=interps['feature'][0], edge_index1=gnn_info[0]['edges'].T, edge_attr1=gnn_info[0]['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos1=gnn_info[0]['coords'], feature2=interps['feature'][1], edge_index2=gnn_info[1]['edges'].T, edge_attr2=gnn_info[1]['edge_weights'] if self.kwargs['gnn'].get('use_edge_attr', True) else None, pos2=gnn_info[1]['coords'])
            else:
                raise NotImplementedError('Only GAT is supported for multi-scale')
            pred_gnn_1 = torch.sigmoid(pred_gnn_1).squeeze(-1)
            pred_gnn_2 = torch.sigmoid(pred_gnn_2).squeeze(-1)
            bceloss1 = nn.BCELoss(reduction='none')
            loss_type = gnn_conf.get('loss_type', 'focal')
            if loss_type == 'focal':
                bceloss0 = BinaryFocalLoss(neg_weight=1, gamma=2, reduction='none')
            elif loss_type == 'dice':
                bceloss0 = DiceLoss1d()
            elif loss_type == 'bce':
                bceloss0 = BCELossPosWeight(pos_weight=1, reduction='none')
            loss_gnn_1 = bceloss0(pred_gnn_1, interps['gt'][0])
            loss_gnn_2 = bceloss0(pred_gnn_2, interps['gt'][1])
            if self.kwargs.get('frangi_in_pseudo') and self.kwargs['frangi_in_pseudo']['use']:
                if gnn_conf.get('use_frangi', True):
                    if is_hard_frangi:
                        loss_gnn_l1_1 = bceloss1(pred_gnn_1, interps['gt'][0])
                        loss_gnn_l1_2 = bceloss1(pred_gnn_2, interps['gt'][1])
                    else:
                        loss_gnn_l1_1 = smoothl1_no_reduction(pred_gnn_1, interps['gt'][0])
                        loss_gnn_l1_2 = smoothl1_no_reduction(pred_gnn_2, interps['gt'][1])
                    loss_gnn_l1_1 = safe_mean_torch(loss_gnn_l1_1, interps['uncertainty'][0] < 0.5) * gnn_conf['weight']
                    loss_gnn_l1_2 = safe_mean_torch(loss_gnn_l1_2, interps['uncertainty'][1] < 0.5) * gnn_conf['weight']
                else:
                    loss_gnn_l1_1 = torch.tensor(0, dtype=x.dtype, device=x.device)
                    loss_gnn_l1_2 = torch.tensor(0, dtype=x.dtype, device=x.device)
                if loss_type not in ['dice']:
                    loss_gnn_1 = safe_mean_torch(loss_gnn_1, interps['uncertainty'][0] > 0.5) * gnn_conf['weight']
                    loss_gnn_2 = safe_mean_torch(loss_gnn_2, interps['uncertainty'][1] > 0.5) * gnn_conf['weight']
                loss_gnn = loss_gnn_1 + loss_gnn_2 + loss_gnn_l1_1 + loss_gnn_l1_2
            elif gnn_conf.get('loss_uncertainty_mask', True):
                loss_gnn_1 = safe_mean_torch(loss_gnn_1, interps['uncertainty'][0] > 0.5) * gnn_conf['weight']
                loss_gnn_2 = safe_mean_torch(loss_gnn_2, interps['uncertainty'][1] > 0.5) * gnn_conf['weight']
                loss_gnn = loss_gnn_1 + loss_gnn_2
            else:
                loss_gnn = loss_gnn_1.mean() * gnn_conf['weight'] + loss_gnn_2.mean() * gnn_conf['weight']
            if gnn_conf.get('consistency'):
                pred_interp_1 = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info[0]['coords'].unsqueeze(0), mode='bilinear')[0].squeeze(-1)
                pred_interp_2 = sample_in_feature_map_3d_torch(pred[:, 1:2], gnn_info[1]['coords'].unsqueeze(0), mode='bilinear')[0].squeeze(-1)
                consistency_loss = smoothl1_loss(pred_interp_1, pred_gnn_1) * gnn_conf['consistency'] + smoothl1_loss(pred_interp_2, pred_gnn_2) * gnn_conf['consistency']
                loss_gnn = loss_gnn + consistency_loss * gnn_conf['weight']
                loss_dic['consistency_loss'] = consistency_loss.item() * gnn_conf['weight']
            if gnn_conf.get('graph_smooth_loss'):
                smooth_1 = graph_smoothness_loss(pred_gnn_1, gnn_info[0]['edges'].T)
                smooth_2 = graph_smoothness_loss(pred_gnn_2, gnn_info[1]['edges'].T)
                loss_gnn = loss_gnn + (smooth_1 + smooth_2) * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
                loss_dic['graph_smooth_loss'] = (smooth_1 + smooth_2).item() * gnn_conf['graph_smooth_loss'] * gnn_conf['weight']
        return (loss_gnn, loss_dic)

    def _calc_pseudo_mask_online(self, x, orig_model):
        x_ori = x.detach().cpu()
        pred_lst = []
        features_lst = []
        for i in range(8):
            with torch.no_grad():
                x_aug = torch.clone(x_ori).to(device=self.device, dtype=torch.float32)
                flip_code = i % 8
                x_aug = flip_3D_torch(x_aug, flip_code)
                pred, leak_dic = orig_model(x_aug)
                pred = flip_3D_torch(pred, flip_code)
                pred = torch.softmax(pred, dim=1)
                pred = pred.detach().cpu()
            pred_lst.append(pred)
            feature = leak_dic['last_feature_map'].detach().cpu()
            feature = flip_3D_torch(feature, flip_code)
            features_lst.append(feature)
        pred_lst = torch.stack(pred_lst)
        pred = pred_lst.mean(dim=0)
        std = pred_lst.std(dim=0)
        max_pos = pred_lst[:, :, 1:2].max(dim=0)[0]
        max_pos = torch.cat([1 - max_pos, max_pos], dim=1)
        write_nii(std[:, 1:2], 'std.nii.gz')
        feature = torch.stack(features_lst).mean(dim=0)
        mask, pseudo_label = get_mask_from_uncertainty(pred, std, feature)
        return (mask, pseudo_label, std)

    def _calc_pseudo_mask_online_from_pred(self, pred_lst, features_lst, and_fn=False):
        check_double_sigmoid(pred_lst)
        pred_lst = torch.sigmoid(pred_lst)
        pred = pred_lst.mean(dim=0, keepdim=True)
        std = pred_lst.std(dim=0, keepdim=True)
        feature = features_lst.mean(dim=0, keepdim=True)
        mask, pseudo_label = get_mask_from_uncertainty(pred, std, feature)
        pseudo_label = pseudo_label[:, 1]
        dtype = pseudo_label.dtype
        mask = mask[:, 1]
        std = std[:, 1]
        mask = 1 - mask
        if not and_fn:
            pseudo_label = (pseudo_label.to(torch.bool) | mask.to(torch.bool)).to(dtype)
        return (pseudo_label, mask, std)

    def _wrap_model_forward_eval(self, **kwargs):
        if kwargs.get('model_to_test'):
            return WrappedModel(self.models_dict[kwargs['model_to_test']])
        else:
            return WrappedModel(self.model)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='configs/exp_gnn_pseudo_learn/tof_to_snap.yaml')
    return args.parse_args()

def choose_dataset(full_config):
    root = full_config.get('data_root', 'data/SNAP')
    other = {k: full_config[k] for k in ['pseudo_path', 'mask_path', 'wanted_names'] if k in full_config}
    return (root, full_config.get('invert_grayscale', True), full_config.get('norm_method', {'name': 'znorm'}),
            full_config.get('frangi_path', osp.join(root, 'frangi', 'imagesTr')), (12, 24), other)


def main():
    args = parse_args()
    full_config = load_yaml(args.config)
    configure_runtime(full_config)
    device = torch.device(full_config.get('device', 'cuda:0'))
    is_train_refiner = full_config.get('train_refiner', False)
    patch_size_description = full_config['trainer_configs'].get('patch_size_description')
    if patch_size_description == 'large':
        full_config['trainer_configs']['patch_size'] = [64, 320, 256]
        print('patch size set to large: (64, 320, 256)')
    elif patch_size_description == 'small':
        full_config['trainer_configs']['patch_size'] = [40, 224, 192]
        print('patch size set to small: (40, 224, 192)')
    else:
        print(f"Using custom patch size {full_config['trainer_configs']['patch_size']}")
    snap_data_path, invert_grayscale, norm_method, frangi_path, default_graph_side_len, other_dic = choose_dataset(full_config)
    if not full_config.get('graph_version'):
        graph_version = 2
        print(f'graph_version not found in config, using default {graph_version}')
    else:
        graph_version = full_config['graph_version']
    if not full_config.get('highres_graph_sidelen'):
        highres_graph_sidelen = default_graph_side_len[0]
        print(f'highres_graph_sidelen not found in config, using default {highres_graph_sidelen}')
    else:
        highres_graph_sidelen = full_config['highres_graph_sidelen']
    if not full_config.get('lowres_graph_sidelen'):
        lowres_graph_sidelen = default_graph_side_len[1]
        print(f'lowres_graph_sidelen not found in config, using default {lowres_graph_sidelen}')
    else:
        lowres_graph_sidelen = full_config['lowres_graph_sidelen']
    other_aux = [{'name': 'gnn', 'aux_path': snap_data_path + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}', 'type': 'pkl'}]
    if full_config['trainer_configs']['gnn'].get('scales', 1) >= 2:
        other_aux.append({'name': 'gnn2', 'aux_path': snap_data_path + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}', 'type': 'pkl'})
    train_dataset_snap_inv = CustomDataset(snap_data_path, mode='test', test_fd_str='TrMod', invert_grayscale=invert_grayscale, allow_no_label=True, only_no_label=True, last_stage_mask_path={'frangi': frangi_path, 'pseudo': other_dic.get('pseudo_path', None), 'mask': other_dic.get('mask_path', None)}, other_aux=other_aux, load_mask_pseudo=snap_data_path + '/PLPB_mask', norm_method=norm_method, wanted_names=(osp.join(snap_data_path, other_dic['wanted_names'][0]), other_dic['wanted_names'][1]) if 'wanted_names' in other_dic else None)
    test_dataset_snap_inv = CustomDataset(snap_data_path, mode='test', test_fd_str='TrMod', invert_grayscale=invert_grayscale, norm_method=norm_method)
    train_loader_snap_inv = DataLoader(train_dataset_snap_inv, batch_size=1, shuffle=True)
    test_loader_snap_inv = DataLoader(test_dataset_snap_inv, batch_size=1, shuffle=False)
    run_name = full_config.get('run_name', 'vesselgda')
    if not is_train_refiner:
        base_folder = osp.join('results', run_name)
    else:
        conv_layers_trained = full_config['trainer_configs']['refiner']['conv_layers']
        feature_map_trained = full_config['trainer_configs']['refiner']['feature_map_from']
        dataset = full_config['dataset']
        print('Training refiner {} {} {} using {}...'.format(conv_layers_trained, feature_map_trained, patch_size_description, dataset))
        base_folder = 'results/artery_seg_exp_refiner_{}_{}{}_{}'.format(conv_layers_trained, feature_map_trained, '_' + patch_size_description if patch_size_description is not None else '', dataset)
    base_folder = full_config.get('output_dir', base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    write_yaml(join(base_folder, 'config.yaml'), full_config)
    trainer_configs = full_config['trainer_configs']
    if trainer_configs['refine_and_pseudo'] and trainer_configs['reweight_pseudo']:
        raise ValueError('Cannot reweight pseudo label when refine_and_pseudo is True')
    training_configs = full_config['training_configs']
    trainer_configs.update(training_configs)
    model = make_net(net_conf=patch_size_description if patch_size_description is not None else 'large', track=full_config.get('track', False))
    if training_configs['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    elif training_configs['optimizer_type'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    elif training_configs['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], momentum=0.9)
    elif training_configs['optimizer_type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    model = model.to(device)
    criterion = None
    trainer = TrainerGNN(model, optimizer, criterion, device, train_loader_snap_inv, test_loader_snap_inv, base_folder=base_folder, set_to_eval_when_testing=True, **trainer_configs)
    if is_train_refiner:
        trainer.train(val_interval=None, trained_network='refiner')
    else:
        trainer.orig_model.eval()
        trainer.teacher_model.eval()
        trainer.train()
        trainer.test(test_loader_snap_inv, 'SnapInv')
        if full_config['trainer_configs'].get('training_base') == 'aif_sfda':
            raise NotImplementedError('This optional baseline experiment is outside this release.')
        if full_config['trainer_configs'].get('ema'):
            trainer.test(test_loader_snap_inv, 'SnapInv', model_to_test='original')
if __name__ == '__main__':
    main()
