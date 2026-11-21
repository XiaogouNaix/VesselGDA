import torch
import numpy as np
from medpy.metric import hd95, assd, asd
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def dice_torch(pred, target, smooth=1e-06, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    elif len(pred.shape) == 4:
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    dice = (2.0 * intersection + smooth) / (union + smooth)
    if return_std:
        return (dice.mean().item(), dice.std().item())
    else:
        return dice.mean().item()

def dice_np(pred, target, smooth=1e-06):
    assert np.all(pred.shape == target.shape)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def iou_torch(pred, target, smooth=1e-06, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) - intersection
    elif len(pred.shape) == 4:
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    iou = (intersection + smooth) / (union + smooth)
    if return_std:
        return (iou.mean().item(), iou.std().item())
    else:
        return iou.mean().item()

def iou_np(pred, target, smooth=1e-06):
    assert np.all(pred.shape == target.shape)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def precision_recall_torch(pred, target, smooth=1e-06, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        intersection = (pred * target).sum(dim=(2, 3, 4))
        precision = intersection / (pred.sum(dim=(2, 3, 4)) + smooth)
        recall = intersection / (target.sum(dim=(2, 3, 4)) + smooth)
    elif len(pred.shape) == 4:
        intersection = (pred * target).sum(dim=(2, 3))
        precision = intersection / (pred.sum(dim=(2, 3)) + smooth)
        recall = intersection / (target.sum(dim=(2, 3)) + smooth)
    if return_std:
        return (precision.mean().item(), recall.mean().item(), precision.std().item(), recall.std().item())
    else:
        return (precision.mean().item(), recall.mean().item())

def precision_recall_np(pred, target, smooth=1e-06):
    assert np.all(pred.shape == target.shape)
    intersection = (pred * target).sum()
    precision = intersection / (pred.sum() + smooth)
    recall = intersection / (target.sum() + smooth)
    return (precision, recall)

def hd95_np(pred, target, spacing=None):
    assert np.all(pred.shape == target.shape)
    hd = hd95(pred, target, voxelspacing=spacing, connectivity=1)
    return hd

def hd95_torch(pred, target, spacing=None, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    elif len(pred.shape) == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    total_hd95 = []
    for i in range(pred.shape[0]):
        total_hd95.append(hd95_np(pred_np[i], target_np[i], spacing=spacing))
    total_hd95_mean = np.mean(total_hd95)
    if return_std:
        total_hd95_std = np.std(total_hd95)
        return (total_hd95_mean, total_hd95_std)
    else:
        return total_hd95_mean

def assd_np(pred, target, spacing=None):
    assert np.all(pred.shape == target.shape)
    return assd(pred, target, voxelspacing=spacing)

def assd_torch(pred, target, spacing=None, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    elif len(pred.shape) == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    total_assd = []
    for i in range(pred.shape[0]):
        total_assd.append(assd_np(pred_np[i], target_np[i], spacing=spacing))
    total_assd_mean = np.mean(total_assd)
    if return_std:
        total_assd_std = np.std(total_assd)
        return (total_assd_mean, total_assd_std)
    else:
        return total_assd_mean

def asd_np(pred, target, spacing=None):
    assert np.all(pred.shape == target.shape)
    return asd(pred, target, voxelspacing=spacing)

def asd_torch(pred, target, spacing=None, return_std=False):
    pred = pred.contiguous()
    target = target.contiguous()
    assert pred.shape[1] == 1
    assert pred.shape == target.shape
    if len(pred.shape) == 5:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    elif len(pred.shape) == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    total_asd = []
    for i in range(pred.shape[0]):
        total_asd.append(asd_np(pred_np[i], target_np[i], spacing=spacing))
    total_asd_mean = np.mean(total_asd)
    if return_std:
        total_asd_std = np.std(total_asd)
        return (total_asd_mean, total_asd_std)
    return total_asd_mean

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / np.sum(s)

def clDice_np(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec + tsens)

def clDice_torch(v_p, v_l, return_std=False):
    assert v_p.shape == v_l.shape, 'Predicted and target tensors must have the same shape.'
    v_p = v_p.cpu().numpy()
    v_l = v_l.cpu().numpy()
    assert v_p.shape[1] == 1, 'Input tensors must have a single channel.'
    if len(v_p.shape) == 5:
        v_p = v_p.squeeze(1)
        v_l = v_l.squeeze(1)
    elif len(v_p.shape) == 4:
        v_p = v_p.squeeze(1)
        v_l = v_l.squeeze(1)
    else:
        raise ValueError('Input tensors must be 4D or 5D (batch, channels, height, width, depth).')
    cldice_scores = []
    for i in range(v_p.shape[0]):
        cldice_scores.append(clDice_np(v_p[i], v_l[i]))
    if return_std:
        return (np.mean(cldice_scores), np.std(cldice_scores))
    return np.mean(cldice_scores)
