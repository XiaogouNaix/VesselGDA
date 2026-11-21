import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchio as tio
from utils import get_center_value_3d, select_voxels_by_mask_3d
from utils_torch_versions_of_funcs import GPULabelSampler
from custom_loss import KLDivLossNormal, WassersteinLossNormal

class RandomContrastiveLoss(nn.Module):

    def __init__(self, num_fg_blocks, num_bg_blocks, crop_size=(8, 20, 20), loss_type='normal', eps=1e-05, margin=2):
        super(RandomContrastiveLoss, self).__init__()
        self.eps = eps
        self.num_fg_blocks = num_fg_blocks
        self.num_bg_blocks = num_bg_blocks
        self.crop_size = crop_size
        self.kl_type = loss_type
        self.margin = margin
        assert loss_type in ['normal', 'instance', 'wass', 'cosine']
        if loss_type == 'normal':
            self.kl_loss = KLDivLossNormal(reduction='mean')
        elif loss_type == 'wass':
            self.kl_loss = WassersteinLossNormal(reduction='mean')
        elif loss_type == 'instance':
            self.kl_loss = nn.KLDivLoss(reduction='mean')
        elif loss_type == 'cosine':
            self.kl_loss = nn.CosineEmbeddingLoss(reduction='mean')
        fg_prob = num_fg_blocks / (num_fg_blocks + num_bg_blocks + 1e-06)
        self.sampler = tio.sampler.LabelSampler(patch_size=crop_size, label_name='label', label_probabilities={0: 1 - fg_prob, 1: fg_prob})

    def forward(self, feature, pseudo, mask):
        pseudo = F.interpolate(pseudo, size=feature.shape[2:], mode='nearest')
        mask = F.interpolate(mask, size=feature.shape[2:], mode='nearest')
        label_for_fg = pseudo * mask
        label_for_bg = (1 - pseudo) * mask
        device = feature.device
        feature = feature.cpu()
        pseudo = pseudo.cpu()
        mask = mask.cpu()
        label_for_fg = label_for_fg.cpu()
        label_for_bg = label_for_bg.cpu()
        subject = tio.Subject(image=tio.ScalarImage(tensor=feature.squeeze(0)), label=tio.LabelMap(tensor=label_for_fg.squeeze(0)), pseudo=tio.ScalarImage(tensor=pseudo.squeeze(0)), mask=tio.ScalarImage(tensor=mask.squeeze(0)))
        sampler = self.sampler(subject, self.num_fg_blocks + self.num_bg_blocks)
        positive_tensors_mean = []
        negative_tensors_mean = []
        positive_tensors_std = []
        negative_tensors_std = []
        for sample in sampler:
            if get_center_value_3d(sample['label']['data'].squeeze(0)) == 1:
                pos_mean, pos_std = self.process_tensor(sample['image']['data'], sample['pseudo']['data'], sample['mask']['data'], is_fg=True, device=device)
                positive_tensors_mean.append(pos_mean)
                positive_tensors_std.append(pos_std)
            else:
                neg_mean, neg_std = self.process_tensor(sample['image']['data'], sample['pseudo']['data'], sample['mask']['data'], is_fg=False, device=device)
                negative_tensors_mean.append(neg_mean)
                negative_tensors_std.append(neg_std)
        loss_all = 0
        loss_pos1 = self._do_calculate_loss(positive_tensors_mean, positive_tensors_std, positive_tensors_mean, positive_tensors_std)
        loss_neg = self._do_calculate_loss(positive_tensors_mean, positive_tensors_std, negative_tensors_mean, negative_tensors_std)
        loss_pos2 = self._do_calculate_loss(negative_tensors_mean, negative_tensors_std, negative_tensors_mean, negative_tensors_std)
        if loss_neg > self.margin:
            loss_neg = self.margin
        loss_all += loss_pos1 + loss_pos2 - loss_neg
        return loss_all

    def _do_calculate_loss(self, mean1, std1, mean2, std2):
        loss_all = 0
        for pm, ps in zip(mean1, std1):
            for nm, ns in zip(mean2, std2):
                if self.kl_type == 'normal':
                    loss = (self.kl_loss(pm, ps, nm, ns) + self.kl_loss(nm, ns, pm, ps)) / 2
                elif self.kl_type == 'wass':
                    loss = self.kl_loss(pm, ps, nm, ns)
                elif self.kl_type == 'instance':
                    pm_sig = torch.sigmoid(pm)
                    nm_sig = torch.sigmoid(nm)
                    loss = (self.kl_loss(torch.log(pm_sig), nm_sig) + self.kl_loss(torch.log(nm_sig), pm_sig)) / 2
                elif self.kl_type == 'cosine':
                    loss = self.kl_loss(pm.unsqueeze(0), nm.unsqueeze(0), torch.ones(1).to(pm.device))
                loss_all += loss
        loss_all = loss_all / (len(mean1) * len(mean2))
        return loss_all

    def process_tensor(self, image, pseudo, mask, is_fg, device):
        image = image.to(device)
        pseudo = pseudo.to(device)
        mask = mask.to(device)
        if is_fg:
            wanted_mask = pseudo * mask
        else:
            wanted_mask = (1 - pseudo) * mask
        processed = select_voxels_by_mask_3d(image, wanted_mask.squeeze(0) == 1)
        processed_mean = torch.mean(processed, dim=1)
        processed_std = torch.std(processed, dim=1)
        return (processed_mean, processed_std)

class RandomContrastiveLossMoco(RandomContrastiveLoss):

    def __init__(self, num_fg_blocks, num_bg_blocks, crop_size=(8, 20, 20), eps=1e-05, margin=2, temperature=0.07):
        super(RandomContrastiveLoss, self).__init__()
        self.eps = eps
        self.num_fg_blocks = num_fg_blocks
        self.num_bg_blocks = num_bg_blocks
        self.crop_size = crop_size
        self.margin = margin
        self.temperature = temperature
        fg_prob = num_fg_blocks / (num_fg_blocks + num_bg_blocks + 1e-06)
        self.sampler = tio.sampler.LabelSampler(patch_size=crop_size, label_name='label', label_probabilities={0: 1 - fg_prob, 1: fg_prob})

    def forward(self, feature, pseudo, mask):
        pseudo = F.interpolate(pseudo, size=feature.shape[2:], mode='nearest')
        mask = F.interpolate(mask, size=feature.shape[2:], mode='nearest')
        label_for_fg = pseudo * mask
        label_for_bg = (1 - pseudo) * mask
        device = feature.device
        feature = feature.cpu()
        pseudo = pseudo.cpu()
        mask = mask.cpu()
        label_for_fg = label_for_fg.cpu()
        label_for_bg = label_for_bg.cpu()
        subject = tio.Subject(image=tio.ScalarImage(tensor=feature.squeeze(0)), label=tio.LabelMap(tensor=label_for_fg.squeeze(0)), pseudo=tio.ScalarImage(tensor=pseudo.squeeze(0)), mask=tio.ScalarImage(tensor=mask.squeeze(0)))
        sampler = self.sampler(subject, self.num_fg_blocks + self.num_bg_blocks)
        positive_tensors_mean = []
        negative_tensors_mean = []
        positive_tensors_std = []
        negative_tensors_std = []
        for sample in sampler:
            if get_center_value_3d(sample['label']['data'].squeeze(0)) == 1:
                pos_mean, pos_std = self.process_tensor(sample['image']['data'], sample['pseudo']['data'], sample['mask']['data'], is_fg=True, device=device)
                positive_tensors_mean.append(pos_mean)
                positive_tensors_std.append(pos_std)
            else:
                neg_mean, neg_std = self.process_tensor(sample['image']['data'], sample['pseudo']['data'], sample['mask']['data'], is_fg=False, device=device)
                negative_tensors_mean.append(neg_mean)
                negative_tensors_std.append(neg_std)
        loss_all = 0
        for p in positive_tensors_mean:
            nominator = 0
            denominator = 0
            for n in negative_tensors_mean:
                denominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), n.unsqueeze(0), dim=1) / self.temperature)
            for pp in positive_tensors_mean:
                if torch.all(torch.eq(p, pp)):
                    continue
                nominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), pp.unsqueeze(0), dim=1) / self.temperature)
                denominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), pp.unsqueeze(0), dim=1) / self.temperature)
            loss_all += -torch.log(nominator / denominator)
        for n in negative_tensors_mean:
            nominator = 0
            denominator = 0
            for nn in negative_tensors_mean:
                if torch.all(torch.eq(n, nn)):
                    continue
                nominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), nn.unsqueeze(0), dim=1) / self.temperature)
                denominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), nn.unsqueeze(0), dim=1) / self.temperature)
            for p in positive_tensors_mean:
                denominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), p.unsqueeze(0), dim=1) / self.temperature)
            loss_all += -torch.log(nominator / denominator)
        return loss_all

class AugRandomContrastiveLossMoco(RandomContrastiveLoss):

    def __init__(self, num_fg_blocks, num_bg_blocks, crop_size=(8, 20, 20), eps=1e-05, margin=2, temperature=0.07, uncertain_prob=0, wanted_features=['mean'], dense_contrastive_loss={'use': False, 'weight': 1.0, 'mode': 'dense', 'sample_num': 7, 'part1_weight': 1.0, 'part2_weight': 1.0}, aug_l1_loss={'use': False, 'weight': 1, 'fg_weight': 3, 'uncertain_higher_weight': False}, moco_weight=1.0):
        super(RandomContrastiveLoss, self).__init__()
        self.eps = eps
        self.num_fg_blocks = num_fg_blocks
        self.num_bg_blocks = num_bg_blocks
        self.crop_size = crop_size
        self.margin = margin
        self.temperature = temperature
        self.uncertain_prob = uncertain_prob
        self.wanted_features = wanted_features
        self.dense_contrastive_loss = dense_contrastive_loss
        self.aug_l1_loss = aug_l1_loss
        self.moco_weight = moco_weight
        fg_prob = num_fg_blocks / (num_fg_blocks + num_bg_blocks + 1e-06)
        self.sampler = GPULabelSampler(patch_size=crop_size, samples_per_volume=num_bg_blocks + num_fg_blocks, prob_map='weighted_custom', prob_weights={0: (1 - fg_prob) * (1 - uncertain_prob), 1: fg_prob * (1 - uncertain_prob), 2: uncertain_prob}, max_iter=20)
        self.get_center_value = get_center_value_3d
        self.select_voxels_by_mask = select_voxels_by_mask_3d

    def forward(self, feature, feature_aug, pseudo, mask, **kwargs):
        feature, feature_aug, pseudo, mask = (self._pre_operation(feature), self._pre_operation(feature_aug), self._pre_operation(pseudo), self._pre_operation(mask))
        pseudo = F.interpolate(pseudo.float(), size=feature.shape[2:], mode='nearest')
        mask = F.interpolate(mask, size=feature.shape[2:], mode='nearest')
        if 'entropy_map' in kwargs:
            kwargs['entropy_map'] = self._pre_operation(kwargs['entropy_map'])
            kwargs['entropy_map'] = F.interpolate(kwargs['entropy_map'], size=feature.shape[2:], mode='nearest').cpu()
        fg_certain = pseudo * mask
        bg_certain = (1 - pseudo) * mask
        fg_uncertain = pseudo * (1 - mask)
        bg_uncertain = (1 - pseudo) * (1 - mask)
        device = feature.device
        label_map = fg_certain + (fg_uncertain + bg_uncertain) * 2
        subject_dic = {'image': feature[0].clone(), 'image_aug': feature_aug[0].clone(), 'label': label_map[0].clone(), 'pseudo': pseudo[0].clone(), 'mask': mask[0].clone()}
        if 'entropy_map' in kwargs:
            subject_dic['entropy_map'] = kwargs['entropy_map'][0].clone().to(device=device)
        label_for_sampler = label_map[0]
        samples = self.sampler(subject_dic, label_for_sampler)
        positive_tensors_mean = []
        positive_aug_tensors_mean = []
        negative_tensors_mean = []
        negative_aug_tensors_mean = []
        positive_tensors = []
        positive_aug_tensors = []
        negative_tensors = []
        negative_aug_tensors = []
        loss_l1 = 0
        if self.aug_l1_loss['use'] and (not self.aug_l1_loss.get('patched', True)):
            loss_l1 = self._calc_aug_l1_loss(feature, feature_aug, pseudo, kwargs['entropy_map'])
        for idx in range(len(samples['image'])):
            if self.aug_l1_loss['use'] and self.aug_l1_loss.get('patched', True):
                loss_l1 = loss_l1 + self._calc_aug_l1_loss(samples['image'][idx], samples['image_aug'][idx], samples['pseudo'][idx], samples['entropy_map'][idx])
            if self.get_center_value(samples['label'][idx].squeeze(0)) == 1:
                pos_mean = self.process_tensor(samples['image'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=True, device=device)
                if pos_mean is None:
                    continue
                pos_aug_mean = self.process_tensor(samples['image_aug'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=True, device=device)
                positive_tensors_mean.append(pos_mean)
                positive_aug_tensors_mean.append(pos_aug_mean)
                if self.dense_contrastive_loss['use']:
                    pos_tensor = self.process_tensor(samples['image'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=True, device=device, return_original_processed=True)
                    pos_aug_tensor = self.process_tensor(samples['image_aug'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=True, device=device, return_original_processed=True)
                    positive_tensors.append(pos_tensor)
                    positive_aug_tensors.append(pos_aug_tensor)
                    neg_tensor = self.process_tensor(samples['image'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=False, device=device, return_original_processed=True)
                    neg_aug_tensor = self.process_tensor(samples['image_aug'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=False, device=device, return_original_processed=True)
                    negative_tensors.append(neg_tensor)
                    negative_aug_tensors.append(neg_aug_tensor)
            else:
                neg_mean = self.process_tensor(samples['image'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=False, device=device)
                if neg_mean is None:
                    continue
                neg_aug_mean = self.process_tensor(samples['image_aug'][idx], samples['pseudo'][idx], samples['mask'][idx], is_fg=False, device=device)
                negative_tensors_mean.append(neg_mean)
                negative_aug_tensors_mean.append(neg_aug_mean)
        loss_all = 0
        if self.moco_weight > 0:
            for p, pp in zip(positive_tensors_mean, positive_aug_tensors_mean):
                nominator = 0
                denominator = 0
                for n in negative_tensors_mean:
                    denominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), n.unsqueeze(0), dim=1) / self.temperature)
                nominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), pp.unsqueeze(0), dim=1) / self.temperature)
                denominator += torch.exp(torch.cosine_similarity(p.unsqueeze(0), pp.unsqueeze(0), dim=1) / self.temperature)
                loss_all += -torch.log(nominator / denominator)
            for n, nnn in zip(negative_tensors_mean, negative_aug_tensors_mean):
                nominator = 0
                denominator = 0
                for p in positive_tensors_mean:
                    denominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), p.unsqueeze(0), dim=1) / self.temperature)
                nominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), nnn.unsqueeze(0), dim=1) / self.temperature)
                denominator += torch.exp(torch.cosine_similarity(n.unsqueeze(0), nnn.unsqueeze(0), dim=1) / self.temperature)
                loss_all += -torch.log(nominator / denominator)
            loss_all = loss_all * self.moco_weight
        if self.dense_contrastive_loss['use']:
            loss_dense = self._calc_dense_contrastive_loss(positive_tensors, negative_tensors, positive_aug_tensors, negative_aug_tensors)
            loss_all += loss_dense
        if self.aug_l1_loss['use']:
            loss_all += loss_l1 * self.aug_l1_loss['weight']
        return loss_all

    def _sample_to_cuda(self, sample, device):
        for key in ['image', 'image_aug', 'label', 'pseudo', 'mask', 'entropy_map']:
            if key in sample:
                sample[key] = sample[key].to(device)

    def process_tensor(self, image, pseudo, mask, is_fg, device, return_original_processed=False):
        pseudo[pseudo > 0.5] = 1
        pseudo[pseudo <= 0.5] = 0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image = image.to(device)
        pseudo = pseudo.to(device)
        mask = mask.to(device)
        if is_fg:
            wanted_mask = pseudo * mask
        else:
            wanted_mask = (1 - pseudo) * mask
        if torch.sum(wanted_mask) == 0:
            return None
        processed = self.select_voxels_by_mask(image, wanted_mask.squeeze(0) == 1)
        if return_original_processed:
            return processed.permute(1, 0)
        processed_dic = {}
        processed_dic['mean'] = torch.mean(processed, dim=1)
        processed_dic['std'] = torch.std(processed, dim=1)
        processed_dic['max'] = torch.max(processed, dim=1)[0]
        final_tensor = torch.tensor([]).to(device)
        for key in self.wanted_features:
            final_tensor = torch.cat((final_tensor, processed_dic[key]), dim=0)
        return final_tensor

    def _calc_dense_contrastive_loss(self, positive_tensor, negative_tensor, positive_aug_tensor, negative_aug_tensor):
        loss_all = 0
        cos = nn.CosineSimilarity(dim=1, eps=1e-06)
        num_valid = 1e-08
        for p, pp, n, nnn in zip(positive_tensor, positive_aug_tensor, negative_tensor, negative_aug_tensor):
            if p.shape[0] == 0 or n.shape[0] == 0:
                continue
            num_valid += 1
            if self.dense_contrastive_loss['mode'] == 'sample':
                p0 = self._sample_tensor(p, self.dense_contrastive_loss['sample_num'])
                n0 = self._sample_tensor(n, self.dense_contrastive_loss['sample_num'])
            else:
                p0 = p
                n0 = n
            p1 = p0.unsqueeze(2)
            p2 = p1.permute(2, 1, 0)
            n1 = n0.unsqueeze(2)
            n2 = n1.permute(2, 1, 0)
            positive_loss_exp = torch.exp(cos(p1, p2) / self.temperature)
            negative_loss_exp = torch.exp(cos(p1, n2) / self.temperature)
            negative_loss_exp_sum = torch.sum(negative_loss_exp, dim=1, keepdim=True)
            total_log_sum = torch.log(positive_loss_exp / (positive_loss_exp + negative_loss_exp_sum))
            total_log_sum = -torch.mean(total_log_sum)
            loss_all = loss_all + total_log_sum * self.dense_contrastive_loss['part1_weight']
        loss_all = loss_all / num_valid
        loss_all = loss_all * self.dense_contrastive_loss['weight']
        return loss_all

    def _calc_aug_l1_loss(self, map, map_aug, pseudo, entropy_map):
        weight_map_ent = torch.exp(entropy_map * float(self.aug_l1_loss['uncertain_higher_weight']))
        weight_map_pseudo = torch.ones_like(pseudo)
        weight_map_pseudo[pseudo == 1] = self.aug_l1_loss['fg_weight']
        weight_map = weight_map_ent * weight_map_pseudo
        loss_function = nn.SmoothL1Loss(reduction='none')
        loss = loss_function(map, map_aug) * weight_map
        loss = loss.mean() / weight_map.mean()
        return loss

    def _sample_tensor(self, tensor, num_sample):
        indices = np.random.choice(tensor.shape[0], num_sample)
        sampled_tensor = tensor[indices]
        return sampled_tensor

    def _set_diagonal_to_zero(self, tensor):
        assert tensor.shape[0] == tensor.shape[1], 'Tensor must be square'
        assert len(tensor.shape) == 2, 'Tensor must be 2D'
        for i in range(tensor.shape[0]):
            tensor[i, i] = 0
        return tensor

    def _pre_operation(self, tensor):
        return tensor

class AugRandomContrastiveLossMoco2d(AugRandomContrastiveLossMoco):

    def __init__(self, num_fg_blocks, num_bg_blocks, crop_size=(20, 20), eps=1e-05, margin=2, temperature=0.07, uncertain_prob=0, wanted_features=['mean'], dense_contrastive_loss={'use': False, 'weight': 1.0, 'simple_weight': 1.0, 'mode': 'dense', 'sample_num': 7, 'part1_weight': 1.0, 'part2_weight': 1.0}, aug_l1_loss={'use': False, 'weight': 1, 'fg_weight': 3, 'uncertain_higher_weight': False}, moco_weight=1.0):
        crop_size_1 = (1, crop_size[0], crop_size[1])
        super(AugRandomContrastiveLossMoco2d, self).__init__(num_fg_blocks, num_bg_blocks, crop_size=crop_size_1, eps=eps, margin=margin, temperature=temperature, uncertain_prob=uncertain_prob, wanted_features=wanted_features, dense_contrastive_loss=dense_contrastive_loss, aug_l1_loss=aug_l1_loss, moco_weight=moco_weight)
        self.get_center_value = get_center_value_3d
        self.select_voxels_by_mask = select_voxels_by_mask_3d

    def process_tensor(self, image, pseudo, mask, is_fg, device, return_original_processed=False):
        return super().process_tensor(image, pseudo, mask, is_fg, device, return_original_processed)

    def _pre_operation(self, tensor):
        tensor = tensor.unsqueeze(2).to(dtype=torch.float32)
        return tensor

def feature_map_consistency_loss(feature1, feature2, weight_map, weight_map_scale=1.0):
    weight_map = F.interpolate(weight_map, size=feature1.shape[2:], mode='bilinear')
    diff = F.smooth_l1_loss(feature1, feature2, reduction='none')
    weighted_diff = diff * torch.exp(weight_map_scale * weight_map)
    loss = torch.mean(weighted_diff)
    return loss
