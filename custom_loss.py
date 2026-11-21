import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def skeletonize_torch(tensor):
    assert tensor.shape[1] == 1, 'Num of channels must be 1'
    skel_lst = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i, 0].cpu().numpy()
        skel_np = skeletonize(img_np).astype(np.float32)
        skel_tensor = torch.from_numpy(skel_np).to(device=tensor.device, dtype=tensor.dtype)
        skel_lst.append(skel_tensor.unsqueeze(0))
    skel_tensor = torch.cat(skel_lst, dim=0).unsqueeze(1)
    return skel_tensor

def dice(output, target, eps=0.1):
    num = 2 * (output * target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den

def masked_dice(output, target, mask, eps=0.1):
    num = 2 * (output * target * mask).sum() + eps
    den = (mask * (output + target)).sum() + eps
    return 1.0 - num / den

def masked_cross_entropy_dice(output, target, mask, weight=1.0, weight_dice=1.0):
    loss = weight * F.cross_entropy(output, target, reduction='none') * mask.float()
    loss = loss.mean() / mask.float().mean()
    if weight_dice > 0:
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=output.shape[1]).permute(0, 4, 1, 2, 3)
        for b in range(output.shape[0]):
            mask_for_dice_this = mask.float()[b][0]
            for c in range(output.shape[1]):
                o = output[b][c]
                t = target[b][c].float()
                loss += 0.25 * dice(o * mask_for_dice_this, t * mask_for_dice_this) * weight_dice
    return loss

def masked_bce_with_logits_dice(output, target, mask, weight=1.0, weight_dice=1.0, uncertain_weight=0.0, uncertain_weight_dice=0.0):
    loss_map = weight * F.binary_cross_entropy_with_logits(output, target, reduction='none')
    loss = loss_map * mask.float()
    loss = loss.mean() / (mask.float().mean() + 1e-06)
    if uncertain_weight > 0:
        loss_uncertain = loss_map * (1 - mask).float()
        loss += uncertain_weight * weight * loss_uncertain.mean() / ((1 - mask).float().mean() + 1e-06)
    if weight_dice > 0:
        output = torch.sigmoid(output)
        loss_dice = 0.0
        for n in range(output.shape[0]):
            for c in range(output.shape[1]):
                o = output[n][c]
                t = target[n][c].float()
                if uncertain_weight_dice <= 0:
                    mask_for_dice = mask[n][c]
                    loss_dice += 0.25 * dice(o * mask_for_dice, t * mask_for_dice) * weight_dice
                else:
                    mask_for_dice = mask[n][c] + (1 - mask[n][c]) * uncertain_weight_dice
                    loss_dice += 0.25 * masked_dice(o, t, mask_for_dice)
        loss += loss_dice / output.shape[0]
    return loss

class WeightedDiceLoss(nn.Module):

    def __init__(self, eps=1e-05, sigmoid=False, no_bg=False):
        super().__init__()
        self.eps = eps
        self.sigmoid = sigmoid
        self.no_bg = no_bg

    def forward(self, pred, label, weight=None):
        assert not torch.any(pred == torch.nan)
        assert pred.shape == label.shape
        if pred.shape[1] == 1 or not self.no_bg:
            ch_begin = 0
        else:
            ch_begin = 1
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        r = 0
        if weight is None:
            for p_inst, l_inst in zip(pred, label):
                for j in range(ch_begin, p_inst.shape[0]):
                    A = p_inst[j] * l_inst[j]
                    a = torch.sum(A, dtype=torch.float32) * 2
                    b = torch.sum(p_inst[j], dtype=torch.float32) + torch.sum(l_inst[j], dtype=torch.float32)
                    r1 = (a + self.eps) / (b + self.eps)
                    r += r1
        else:
            for p, l, w in zip(pred, label, weight):
                for j in range(ch_begin, p.shape[0]):
                    A = p[j] * l[j]
                    a = torch.sum(A, dtype=torch.float32) * 2
                    b = torch.sum(p[j], dtype=torch.float32) + torch.sum(l[j], dtype=torch.float32)
                    r1 = (a + self.eps) / (b + self.eps)
                    r1 *= w
                    r += r1
        r /= pred.shape[0] * (pred.shape[1] - ch_begin)
        return 1 - r

class DiceLoss1d(nn.Module):

    def __init__(self, eps=1e-05, sigmoid=False):
        super().__init__()
        self.eps = eps
        self.sigmoid = sigmoid

    def forward(self, pred, label, weight=None):
        assert not torch.any(pred == torch.nan)
        assert pred.shape == label.shape
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)
            label = label.unsqueeze(0)
        nominator = 2 * torch.sum(pred * label, dim=1)
        denominator = torch.sum(pred, dim=1) + torch.sum(label, dim=1)
        r = (nominator + self.eps) / (denominator + self.eps)
        return 1 - r

class BinaryFocalLoss(nn.Module):

    def __init__(self, neg_weight=1, gamma=2, smoothing=0.0, reduction='mean', eps=1e-06):
        """
        Input: pred: [B, 1, ...], label: [B, 1, ...]
        """
        super().__init__()
        self.neg_weight = neg_weight
        self.gamma = gamma
        self.red = reduction
        self.eps = eps

    def forward(self, pred, label, weight=None):
        pos_mask = label
        neg_mask = 1 - label
        pred = torch.clip(pred, self.eps, 1 - self.eps)
        pos_weight = (pos_mask * torch.pow(1 - pred, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(pred)
        neg_weight = (neg_mask * torch.pow(pred, self.gamma)).detach()
        neg_loss = -neg_weight * torch.log(1 - pred) * self.neg_weight
        loss = pos_loss + neg_loss
        if self.red == 'mean':
            loss = loss.mean()
        return loss

class BCELossPosWeight(nn.Module):

    def __init__(self, pos_weight=1.0, reduction='mean', eps=1e-06):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, label):
        pred = torch.clip(pred, self.eps, 1 - self.eps)
        loss = -label * torch.log(pred) * self.pos_weight - (1 - label) * torch.log(1 - pred)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

class KLDivLossNormal(nn.Module):

    def __init__(self, reduction='mean', eps=1e-06):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_mean, pred_std, label_mean, label_std):
        loss = torch.log((label_std + self.eps) / (pred_std + self.eps)) + (torch.pow(pred_std, 2) + torch.pow(pred_mean - label_mean, 2)) / (2 * torch.pow(label_std, 2) + self.eps) - 0.5
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

class WassersteinLossNormal(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_mean, pred_std, label_mean, label_std):
        loss = torch.pow(pred_mean - label_mean, 2) + torch.pow(pred_std - label_std, 2)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

def graph_smoothness_loss(pred, edge_index):
    """
    pred: [N, C]
    edge_index: [2, E]
    """
    row, col = edge_index
    loss = torch.mean((pred[row] - pred[col]) ** 2)
    return loss

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)

class SoftClDice(nn.Module):

    def __init__(self, iter_=3, smooth=1.0, exclude_background=False):
        super(SoftClDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = skeletonize_torch(y_true > 0) * y_true
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice

class SoftClDiceWeighted(nn.Module):

    def __init__(self, iter_=3, smooth=1.0, exclude_background=False, weight_type='inv', power=1.0, eps=1e-06, normalize_weights=True):
        """
        weight_type:
          - "inv": w = 1/(r + eps)
          - "inv_pow": w = 1/(r + eps)**power
          - "exp": w = exp(-power * r)
        power: controls emphasis (for inv_pow or exp)
        normalize_weights: if True, divide weights by their mean (keeps scale stable)
        """
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background
        self.weight_type = weight_type
        self.power = float(power)
        self.eps = float(eps)
        self.normalize_weights = bool(normalize_weights)

    def _make_weight_map_from_mask(self, y_true):
        """
        y_true: tensor, shape (B, C, H, W) or (B, C, D, H, W)
                values are 0/1 or probabilities (we threshold)
        returns weight_map tensor with same shape, dtype/device as y_true
        """
        device = y_true.device
        dtype = y_true.dtype
        y_bin = (y_true > 0).to(dtype=torch.uint8)
        y_np = y_bin.cpu().numpy()
        weight_np = np.zeros_like(y_np, dtype=np.float32)
        it = np.nditer(y_np, flags=['multi_index', 'refs_ok'], op_flags=['readonly'])
        B = y_np.shape[0]
        C = y_np.shape[1]
        for b in range(B):
            for c in range(C):
                mask = y_np[b, c].astype(np.uint8)
                if mask.any():
                    dt = distance_transform_edt(mask)
                    if self.weight_type == 'inv':
                        w = 1.0 / (dt + self.eps)
                    elif self.weight_type == 'inv_pow':
                        w = 1.0 / np.power(dt + self.eps, self.power)
                    elif self.weight_type == 'exp':
                        w = np.exp(-self.power * dt)
                    else:
                        raise ValueError('Unknown weight_type')
                    if self.normalize_weights:
                        mean_w = w[mask.astype(bool)].mean() if mask.any() else 1.0
                        if mean_w > 0:
                            w = w / (mean_w + self.eps)
                    weight_np[b, c] = w
                else:
                    weight_np[b, c] = 0.0
        weight = torch.from_numpy(weight_np).to(device=device, dtype=dtype)
        return weight

    def forward(self, y_true, y_pred):
        """
        y_true: (B, C, H, W) ground truth (0/1 or soft); same convention as your code
        y_pred: (B, C, H, W) predicted probabilities (softmaxed or channel-wise probs)
        """
        if self.exclude_background:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = skeletonize_torch(y_true > 0) * y_true
        with torch.no_grad():
            weight_map = self._make_weight_map_from_mask(y_true)
        numer_tprec = (skel_pred * y_true * weight_map).sum()
        denom_tprec = (skel_pred * weight_map).sum()
        tprec = (numer_tprec + self.smooth) / (denom_tprec + self.smooth)
        numer_tsens = (skel_true * y_pred * weight_map).sum()
        denom_tsens = (skel_true * weight_map).sum()
        tsens = (numer_tsens + self.smooth) / (denom_tsens + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-12)
        return cl_dice
from seg_networks.cbdice_skeletonize import Skeletonize, SoftSkeletonize, combine_tensors, get_weights

class SoftcbDiceLoss(torch.nn.Module):

    def __init__(self, iter_=10, smooth=1.0):
        super(SoftcbDiceLoss, self).__init__()
        self.smooth = smooth
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        """
        Forward pass for the loss function.
        
        Args:
            y_pred (torch.Tensor): Network output with shape (b, c, x, y(, z)).
            y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y(, z)). No one-hot encoding required.
            t_skeletonize_flag (bool, optional): Enable Topology-preserving skeletonization. Defaults to False.
        """
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError('y_true should be 4D or 5D tensor.')
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()
            y_pred_hard = (y_pred_prob > 0.5).float()
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
        skel_pred_prob = skel_pred_hard * y_pred_prob
        q_vl, q_slvl, q_sl = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)
        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl)) + self.smooth) / (torch.sum(combine_tensors(q_spvp, q_slvl, q_sp)) + self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp)) + self.smooth) / (torch.sum(combine_tensors(q_slvl, q_spvp, q_sl)) + self.smooth)
        cb_dice_loss = -2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        return cb_dice_loss

class SymmetricGCE(nn.Module):
    """Symmetric generalized cross-entropy for binary classification."""

    def __init__(self, q=0.7, alpha=1.0, reduction='mean'):
        super(SymmetricGCE, self).__init__()
        self.q = q
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, probs, targets):
        if probs.dim() == 1:
            probs = torch.stack([1 - probs, probs], dim=0 if len(probs.shape) == 1 else 1)
            targets = torch.stack([1 - targets, targets], dim=0 if len(targets.shape) == 1 else 1)
        target_probs = torch.sum(probs * targets, dim=-2)
        if self.q <= 1e-08:
            gce_term = -torch.log(target_probs + 1e-08)
        else:
            gce_term = (1.0 - torch.pow(target_probs, self.q)) / self.q
        if self.alpha > 0:
            pred_probs = torch.sum(targets * torch.log(probs + 1e-08), dim=-2)
            sym_term = -self.alpha * pred_probs
        else:
            sym_term = 0
        losses = gce_term + sym_term
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

class ELRLossOneClass(nn.Module):

    def __init__(self, reduction='mean'):
        super(ELRLossOneClass, self).__init__()
        self.reduction = reduction

    def forward(self, probs, targets):
        probs = torch.stack([1 - probs, probs], dim=0 if len(probs.shape) == 1 else 1)
        targets = torch.stack([1 - targets, targets], dim=0)
        sums = torch.sum(probs * targets, dim=-2)
        loss = torch.log(1 - sums + 1e-08)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
