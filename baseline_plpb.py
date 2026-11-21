import torch
from utils import write_nii

def get_mask_from_uncertainty(pred, std, feature, pred_thres=0.5, std_thres=0.05):
    pseudo_label = pred > pred_thres
    pseudo_label = pseudo_label.float()
    target_0_obj = pseudo_label[:, 0:1]
    target_1_obj = pseudo_label[:, 1:2]
    target_0_bck = 1 - target_0_obj
    target_1_bck = 1 - target_1_obj
    mask_0_obj = (std[:, 0:1] < std_thres).float()
    mask_1_obj = (std[:, 1:2] < std_thres).float()
    mask_0_bck = torch.clone(mask_0_obj)
    mask_1_bck = torch.clone(mask_1_obj)
    mask_0 = mask_0_obj + mask_0_bck
    mask_1 = mask_1_obj + mask_1_bck
    mask = torch.cat([mask_0, mask_1], dim=1)
    feature_0_obj = feature * target_0_obj * mask_0_obj
    feature_1_obj = feature * target_1_obj * mask_1_obj
    feature_0_bck = feature * target_0_bck * mask_0_bck
    feature_1_bck = feature * target_1_bck * mask_1_bck
    centroid_0_obj = torch.sum(feature_0_obj * pred[:, 0:1, ...], dim=[0, 2, 3, 4], keepdim=True)
    centroid_1_obj = torch.sum(feature_1_obj * pred[:, 1:2, ...], dim=[0, 2, 3, 4], keepdim=True)
    centroid_0_bck = torch.sum(feature_0_bck * (1.0 - pred[:, 0:1, ...]), dim=[0, 2, 3, 4], keepdim=True)
    centroid_1_bck = torch.sum(feature_1_bck * (1.0 - pred[:, 1:2, ...]), dim=[0, 2, 3, 4], keepdim=True)
    target_0_obj_cnt = torch.sum(mask_0_obj * target_0_obj * pred[:, 0:1, ...], dim=[0, 2, 3, 4], keepdim=True)
    target_1_obj_cnt = torch.sum(mask_1_obj * target_1_obj * pred[:, 1:2, ...], dim=[0, 2, 3, 4], keepdim=True)
    target_0_bck_cnt = torch.sum(mask_0_bck * target_0_bck * (1.0 - pred[:, 0:1, ...]), dim=[0, 2, 3, 4], keepdim=True)
    target_1_bck_cnt = torch.sum(mask_1_bck * target_1_bck * (1.0 - pred[:, 1:2, ...]), dim=[0, 2, 3, 4], keepdim=True)
    centroid_0_obj = centroid_0_obj / (target_0_obj_cnt + 1e-06)
    centroid_1_obj = centroid_1_obj / (target_1_obj_cnt + 1e-06)
    centroid_0_bck = centroid_0_bck / (target_0_bck_cnt + 1e-06)
    centroid_1_bck = centroid_1_bck / (target_1_bck_cnt + 1e-06)
    distance_0_obj = torch.sum((feature - centroid_0_obj) ** 2, dim=1, keepdim=True)
    distance_0_bck = torch.sum((feature - centroid_0_bck) ** 2, dim=1, keepdim=True)
    distance_1_obj = torch.sum((feature - centroid_1_obj) ** 2, dim=1, keepdim=True)
    distance_1_bck = torch.sum((feature - centroid_1_bck) ** 2, dim=1, keepdim=True)
    proto_pseudo_0 = distance_0_obj < distance_0_bck
    proto_pseudo_1 = distance_1_obj < distance_1_bck
    proto_pseudo = torch.cat([proto_pseudo_0, proto_pseudo_1], dim=1)
    mask_proto = pseudo_label == proto_pseudo
    mask_proto = mask_proto.float()
    mask = mask * mask_proto
    mask = mask / 2
    write_nii(1 - mask[:, 1:2], '1-mask_final.nii.gz')
    return (mask, pseudo_label)

def get_mask_from_uncertainty_2d(pred, std, feature, pred_thres=0.5, std_thres=0.05):
    pseudo_label = pred > pred_thres
    pseudo_label = pseudo_label.float()
    target_0_obj = pseudo_label[:, 0:1]
    target_1_obj = pseudo_label[:, 1:2]
    target_0_bck = 1 - target_0_obj
    target_1_bck = 1 - target_1_obj
    mask_0_obj = (std[:, 0:1] < std_thres).float()
    mask_1_obj = (std[:, 1:2] < std_thres).float()
    mask_0_bck = torch.clone(mask_0_obj)
    mask_1_bck = torch.clone(mask_1_obj)
    mask_0 = mask_0_obj + mask_0_bck
    mask_1 = mask_1_obj + mask_1_bck
    mask = torch.cat([mask_0, mask_1], dim=1)
    feature_0_obj = feature * target_0_obj * mask_0_obj
    feature_1_obj = feature * target_1_obj * mask_1_obj
    feature_0_bck = feature * target_0_bck * mask_0_bck
    feature_1_bck = feature * target_1_bck * mask_1_bck
    centroid_0_obj = torch.sum(feature_0_obj * pred[:, 0:1, ...], dim=[2, 3], keepdim=True)
    centroid_1_obj = torch.sum(feature_1_obj * pred[:, 1:2, ...], dim=[2, 3], keepdim=True)
    centroid_0_bck = torch.sum(feature_0_bck * (1.0 - pred[:, 0:1, ...]), dim=[2, 3], keepdim=True)
    centroid_1_bck = torch.sum(feature_1_bck * (1.0 - pred[:, 1:2, ...]), dim=[2, 3], keepdim=True)
    target_0_obj_cnt = torch.sum(mask_0_obj * target_0_obj * pred[:, 0:1, ...], dim=[2, 3], keepdim=True)
    target_1_obj_cnt = torch.sum(mask_1_obj * target_1_obj * pred[:, 1:2, ...], dim=[2, 3], keepdim=True)
    target_0_bck_cnt = torch.sum(mask_0_bck * target_0_bck * (1.0 - pred[:, 0:1, ...]), dim=[2, 3], keepdim=True)
    target_1_bck_cnt = torch.sum(mask_1_bck * target_1_bck * (1.0 - pred[:, 1:2, ...]), dim=[2, 3], keepdim=True)
    centroid_0_obj = centroid_0_obj / (target_0_obj_cnt + 1e-06)
    centroid_1_obj = centroid_1_obj / (target_1_obj_cnt + 1e-06)
    centroid_0_bck = centroid_0_bck / (target_0_bck_cnt + 1e-06)
    centroid_1_bck = centroid_1_bck / (target_1_bck_cnt + 1e-06)
    distance_0_obj = torch.sum((feature - centroid_0_obj) ** 2, dim=1, keepdim=True)
    distance_0_bck = torch.sum((feature - centroid_0_bck) ** 2, dim=1, keepdim=True)
    distance_1_obj = torch.sum((feature - centroid_1_obj) ** 2, dim=1, keepdim=True)
    distance_1_bck = torch.sum((feature - centroid_1_bck) ** 2, dim=1, keepdim=True)
    proto_pseudo_0 = distance_0_obj < distance_0_bck
    proto_pseudo_1 = distance_1_obj < distance_1_bck
    proto_pseudo = torch.cat([proto_pseudo_0, proto_pseudo_1], dim=1)
    mask_proto = pseudo_label == proto_pseudo
    mask_proto = mask_proto.float()
    mask = mask * mask_proto
    mask = mask / 2
    write_nii(1 - mask[:, 1:2], '1-mask_final.nii.gz')
    return (mask, pseudo_label)
