import numpy as np
import torchio as tio
import torch
import torch.jit
import torch.nn.functional as F
import torchvision.transforms as st_transforms
import torchvision.transforms.functional as TF
from utils_preprocess import flip_3D_torch, rotate_image_cv2, rotate_and_flip_2D_torch
eps = 0.1

def dice(output, target):
    num = 2 * (output * target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den

def cross_entropy_dice(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(output.shape[0]):
        o = output[c]
        t = target[c].float()
        t = t.unsqueeze(0)
        t = F.one_hot(t.to(torch.int64), num_classes=output.shape[1]).permute(0, 4, 1, 2, 3).float()[0]
        loss += 0.25 * dice(o, t)
    return loss

def cross_entropy_dice_2d(output, target, weight=1.0):
    loss = weight * F.binary_cross_entropy_with_logits(output, target)
    output = torch.sigmoid(output)
    for c in range(output.shape[0]):
        o = output[c]
        t = target[c].float()
        loss += 0.25 * dice(o, t)
    return loss

@torch.jit.script
def softmax_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).mean()

@torch.jit.script
def sigmoid_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of sigmoid distribution from logits."""
    x = x.sigmoid()
    return -(x * (x + 1e-06).log() + (1 - x) * (1 + 1e-06 - x).log()).mean()

@torch.jit.script
def entropy_loss_no_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Entropy of sigmoid distribution from logits. Input already done sigmoid."""
    return -(x * (x + 1e-06).log() + (1 - x) * (1 + 1e-06 - x).log()).mean()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(0) * x.log_softmax(0)).sum(0)

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of sigmoid distribution from logits."""
    x = x.sigmoid()
    return -(x * x.log() + (1 - x) * (1 - x).log())

def build_pseduo_augmentation(img):
    aug1 = tio.transforms.RandomMotion()
    aug2 = tio.transforms.RandomGhosting()
    aug3 = tio.transforms.RandomBiasField()
    aug4 = tio.transforms.RandomBlur()
    assert img.shape[0] == 1
    img = img[0]
    aug_img1 = torch.tensor(aug1(img))
    aug_img2 = torch.tensor(aug2(img))
    aug_img3 = torch.tensor(aug3(img))
    aug_img4 = torch.tensor(aug4(img))
    aug_data = torch.cat([torch.tensor(img), aug_img1, aug_img2, aug_img3, aug_img4], dim=0)
    return aug_data.unsqueeze(1)

def build_pseduo_augmentation_plpb(img):
    aug_data_lst = []
    for i in range(8):
        img_aug = torch.from_numpy(img).clone()
        img_aug = flip_3D_torch(img_aug, i)
        aug_data_lst.append(img_aug)
    aug_data = torch.cat(aug_data_lst, dim=0)
    return aug_data

def build_pseduo_augmentation_2d(img):
    if type(img) is np.ndarray:
        img = torch.from_numpy(img).to(dtype=torch.float32)
    aug1 = st_transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)
    aug2 = st_transforms.RandomGrayscale(p=1.0)
    aug3 = st_transforms.RandomGrayscale(p=1.0)
    aug4 = st_transforms.RandomAutocontrast(p=1.0)
    aug_img1 = aug1(img)
    aug_img2 = aug2(img)
    aug_img3 = aug3(img)
    aug_img4 = aug4(img)
    aug_data = torch.stack([img, aug_img1, aug_img2, aug_img3, aug_img4], dim=1)
    return aug_data

def build_pseduo_augmentation_plpb_2d(img):
    aug_data_lst = []
    for j in range(img.shape[0]):
        aug_data_j_lst = []
        for i in range(8):
            angle = i * 45
            img_aug = rotate_image_cv2(np.array(img[j]), angle, background_color=(0, 0, 0))
            img_aug = torch.from_numpy(img_aug).float()
            aug_data_j_lst.append(img_aug)
        aug_data_j = torch.stack(aug_data_j_lst, dim=0)
        aug_data_lst.append(aug_data_j.unsqueeze(0))
    aug_data = torch.cat(aug_data_lst, dim=0)
    return aug_data

def build_pseudo_augmentation_flip_2d(img):
    aug_data_lst = []
    for j in range(img.shape[0]):
        aug_data_j_lst = []
        for i in range(7):
            img_aug = rotate_and_flip_2D_torch(img[j:j + 1], i)
            aug_data_j_lst.append(img_aug)
        aug_data_j = torch.cat(aug_data_j_lst, dim=0)
        aug_data_lst.append(aug_data_j.unsqueeze(0))
    aug_data = torch.cat(aug_data_lst, dim=0)
    return aug_data

def build_pseudo_augmentation_flip_with_gray_2d(img):
    aug_data_lst = []
    C = img.shape[-3]
    for j in range(img.shape[0]):
        aug_data_j_lst = []
        for i in range(7):
            img_aug = rotate_and_flip_2D_torch(img[j:j + 1], i)
            if i % 2 == 1:
                img_aug = TF.rgb_to_grayscale(img_aug, num_output_channels=C)
            aug_data_j_lst.append(img_aug)
        aug_data_j = torch.cat(aug_data_j_lst, dim=0)
        aug_data_lst.append(aug_data_j.unsqueeze(0))
    aug_data = torch.cat(aug_data_lst, dim=0)
    return aug_data

def build_pseudo_augmentation_flip_with_gray_2d_v2(img):
    aug_data_lst = []
    C = img.shape[-3]
    for j in range(img.shape[0]):
        aug_data_j_lst = []
        for i in range(7):
            img_aug = rotate_and_flip_2D_torch(img[j:j + 1], i)
            if i == 3:
                img_aug = TF.rgb_to_grayscale(img_aug, num_output_channels=C)
            aug_data_j_lst.append(img_aug)
        aug_data_j_lst.append(TF.rgb_to_grayscale(img[j:j + 1], num_output_channels=C))
        aug_data_j = torch.cat(aug_data_j_lst, dim=0)
        aug_data_lst.append(aug_data_j.unsqueeze(0))
    aug_data = torch.cat(aug_data_lst, dim=0)
    return aug_data

def flip_back_2d_flip(output):
    assert output.shape[0] == 7
    device = output.device
    output_lst = []
    for i in range(7):
        if 1 <= i <= 3:
            img_back = rotate_and_flip_2D_torch(output[i:i + 1], 4 - i)
        else:
            img_back = rotate_and_flip_2D_torch(output[i:i + 1], i)
        output_lst.append(img_back)
    output = torch.cat(output_lst, dim=0)
    return output.to(device)

def flip_back_2d_flip_with_gray_v2(output):
    assert output.shape[0] == 8
    device = output.device
    output_lst = []
    for i in range(8):
        if 1 <= i <= 3:
            img_back = rotate_and_flip_2D_torch(output[i:i + 1], 4 - i)
        elif 4 <= i <= 6:
            img_back = rotate_and_flip_2D_torch(output[i:i + 1], i)
        else:
            img_back = output[i:i + 1]
        output_lst.append(img_back)
    output = torch.cat(output_lst, dim=0)
    return output.to(device)

def flip_back_plpb(output):
    assert output.shape[0] == 8
    device = output.device
    output = output.detach().cpu()
    output_lst = []
    for i in range(8):
        img_back = flip_3D_torch(output[i:i + 1], i)
        output_lst.append(img_back)
    output = torch.cat(output_lst, dim=0)
    return output.to(device)

def flip_back_plpb_2d(output):
    assert output.shape[0] == 8
    device = output.device
    output = output.detach().cpu()
    output_lst = []
    for i in range(8):
        angle = (8 - i) * 45
        img_back = rotate_image_cv2(np.array(output[i]), angle, background_color='min')
        output_lst.append(img_back)
    output = np.stack(output_lst, axis=0)
    output = torch.from_numpy(output).float()
    return output.to(device)

def uncert_voting(aug_output, and_fn=True):
    device = aug_output.device
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = F.softmax(aug_output[i], dim=0)
        aug_all_prob.append(prob)
        aug_all_ent.append(softmax_entropy(aug_output[i]))
    no_aug_prob_nor = F.softmax(aug_output[0], dim=0)
    no_aug_pseudo_label = no_aug_prob_nor.clone().detach().cpu().numpy()
    no_aug_pseudo_label = no_aug_pseudo_label.argmax(0)
    no_aug_ent = softmax_entropy(aug_output[0])
    no_aug_ent[torch.isnan(no_aug_ent)] = 0
    aug_prob_nor = sum(aug_all_prob) / len(aug_all_prob)
    aug_pseudo_label = aug_prob_nor.clone().detach().cpu().numpy()
    aug_pseudo_label = aug_pseudo_label.argmax(0)
    aug_avg_ent = sum(aug_all_ent) / len(aug_all_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0
    no_aug_ent_nor = (no_aug_ent - no_aug_ent.min()) * (1 / (no_aug_ent.max() - no_aug_ent.min()))
    aug_avg_ent_nor = (aug_avg_ent - aug_avg_ent.min()) * (1 / (aug_avg_ent.max() - aug_avg_ent.min()))
    ent_weight = 0.75
    weighted_ent = ent_weight * no_aug_ent_nor + (1 - ent_weight) * aug_avg_ent_nor
    weighted_ent_thresh = weighted_ent.clone()
    weighted_ent_thresh[weighted_ent_thresh > 0.5] = 1
    weighted_ent_thresh[weighted_ent_thresh <= 0.5] = 0
    prob_min = 0.3
    unct_no_aug_trans = no_aug_prob_nor.clone()
    unct_no_aug_trans[unct_no_aug_trans > 0.5] = 0
    unct_no_aug_trans[unct_no_aug_trans <= prob_min] = 0
    unct_no_aug_trans[unct_no_aug_trans > 0] = 1
    unct_no_aug_prob = unct_no_aug_trans.argmax(0)
    unct_no_aug_prob = torch.tensor(unct_no_aug_prob).clone().detach().to(device)
    weighted_ent_thresh = torch.tensor(weighted_ent_thresh).clone().detach().to(device)
    no_aug_pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device)
    weighted_ent = torch.tensor(weighted_ent).clone().detach().to(device)
    if and_fn:
        pseudo_uncert = unct_no_aug_prob.int() & weighted_ent_thresh.int()
    else:
        pseudo_uncert = weighted_ent_thresh.int()
    pseudo_label = no_aug_pseudo_label.int() | pseudo_uncert.int()
    return (pseudo_label.unsqueeze(0).long(), pseudo_uncert.unsqueeze(0).long(), aug_avg_ent.unsqueeze(0).float())

def uncert_voting_use_pseudo(aug_output, thres=0.15, and_fn=False, uncertain_to_positive=True):
    device = aug_output.device
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = F.softmax(aug_output[i], dim=0)
        aug_all_prob.append(prob)
        aug_all_ent.append(softmax_entropy(aug_output[i]))
    no_aug_prob_nor = F.softmax(aug_output[0], dim=0)
    no_aug_pseudo_label = no_aug_prob_nor.clone().detach().cpu().numpy()
    no_aug_pseudo_label = no_aug_pseudo_label.argmax(0)
    aug_prob_nor = sum(aug_all_prob) / len(aug_all_prob)
    aug_prob_nor = torch.max(aug_prob_nor, dim=0)[0]
    pseudo_uncert = (aug_prob_nor > thres) & (aug_prob_nor < 1 - thres)
    pseudo_uncert = pseudo_uncert.int().unsqueeze(0).long().to(device)
    pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().unsqueeze(0).long().to(device)
    prob_min = 0.3
    unct_no_aug_trans = no_aug_prob_nor.clone()
    unct_no_aug_trans[unct_no_aug_trans > 0.5] = 0
    unct_no_aug_trans[unct_no_aug_trans <= prob_min] = 0
    unct_no_aug_trans[unct_no_aug_trans > 0] = 1
    unct_no_aug_prob = unct_no_aug_trans.argmax(0)
    if and_fn:
        pseudo_uncert = unct_no_aug_prob.int() & pseudo_uncert.int()
    else:
        pseudo_uncert = pseudo_uncert.int()
    aug_avg_ent = sum(aug_all_ent) / len(aug_all_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0
    aug_std = F.softmax(aug_output, dim=0).std(dim=0)
    if uncertain_to_positive:
        pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device) | pseudo_uncert
    else:
        pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device)
    aug_avg_ent = aug_avg_ent.clone().detach().unsqueeze(0).float().to(device)
    return (pseudo_label.long(), pseudo_uncert.long(), aug_std.float()[:, 1])

def uncert_voting_2d(aug_output, and_fn=True, uncertain_to_positive=True):
    device = aug_output.device
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = torch.sigmoid(aug_output[i])
        aug_output_this = torch.cat([1 - aug_output[i], aug_output[i]], dim=0)
        prob = torch.cat([1 - prob, prob], dim=0)
        aug_all_prob.append(prob)
        aug_all_ent.append(sigmoid_entropy(aug_output[i]))
    aug_output_no_aug = torch.sigmoid(aug_output[0])
    aug_output_no_aug = torch.cat([1 - aug_output_no_aug, aug_output_no_aug], dim=0)
    no_aug_prob_nor = F.softmax(aug_output_no_aug, dim=0)
    no_aug_pseudo_label = no_aug_prob_nor.clone().detach().cpu().numpy()
    no_aug_pseudo_label = no_aug_pseudo_label.argmax(0)
    no_aug_ent = sigmoid_entropy(aug_output[0])
    no_aug_ent[torch.isnan(no_aug_ent)] = 0
    aug_prob_nor = sum(aug_all_prob) / len(aug_all_prob)
    aug_pseudo_label = aug_prob_nor.clone().detach().cpu().numpy()
    aug_pseudo_label = aug_pseudo_label.argmax(0)
    aug_avg_ent = sum(aug_all_ent) / len(aug_all_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0
    no_aug_ent_nor = (no_aug_ent - no_aug_ent.min()) * (1 / (no_aug_ent.max() - no_aug_ent.min()))
    aug_avg_ent_nor = (aug_avg_ent - aug_avg_ent.min()) * (1 / (aug_avg_ent.max() - aug_avg_ent.min()))
    ent_weight = 0.75
    weighted_ent = ent_weight * no_aug_ent_nor + (1 - ent_weight) * aug_avg_ent_nor
    weighted_ent_thresh = weighted_ent.clone()
    weighted_ent_thresh[weighted_ent_thresh > 0.5] = 1
    weighted_ent_thresh[weighted_ent_thresh <= 0.5] = 0
    prob_min = 0.3
    unct_no_aug_trans = no_aug_prob_nor.clone()
    unct_no_aug_trans[unct_no_aug_trans > 0.5] = 0
    unct_no_aug_trans[unct_no_aug_trans <= prob_min] = 0
    unct_no_aug_trans[unct_no_aug_trans > 0] = 1
    unct_no_aug_prob = unct_no_aug_trans.argmax(0)
    unct_no_aug_prob = torch.tensor(unct_no_aug_prob).clone().detach().to(device)
    weighted_ent_thresh = torch.tensor(weighted_ent_thresh).clone().detach().to(device)
    no_aug_pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device)
    weighted_ent = torch.tensor(weighted_ent).clone().detach().to(device)
    if and_fn:
        pseudo_uncert = unct_no_aug_prob.int() & weighted_ent_thresh.int()
    else:
        pseudo_uncert = weighted_ent_thresh.int()
    if uncertain_to_positive:
        pseudo_label = no_aug_pseudo_label.int() | pseudo_uncert.int()
    else:
        pseudo_label = no_aug_pseudo_label.int()
    return (pseudo_label.unsqueeze(0).long(), pseudo_uncert.unsqueeze(0).long(), aug_avg_ent.unsqueeze(0).float())

def uncert_voting_2d_use_pseudo(aug_output, thres=0.1, and_fn=False, uncertain_to_positive=True):
    device = aug_output.device
    aug_all_prob = []
    aug_all_ent = []
    for i in range(1, len(aug_output)):
        prob = torch.sigmoid(aug_output[i])
        aug_output_this = torch.cat([1 - aug_output[i], aug_output[i]], dim=0)
        prob = torch.cat([1 - prob, prob], dim=0)
        aug_all_prob.append(prob)
        aug_all_ent.append(sigmoid_entropy(aug_output[i]))
    aug_output_no_aug = torch.sigmoid(aug_output[0])
    aug_output_no_aug = torch.cat([1 - aug_output_no_aug, aug_output_no_aug], dim=0)
    no_aug_prob_nor = F.softmax(aug_output_no_aug, dim=0)
    no_aug_pseudo_label = no_aug_prob_nor.clone().detach().cpu().numpy()
    no_aug_pseudo_label = no_aug_pseudo_label.argmax(0)
    aug_prob_nor = sum(aug_all_prob) / len(aug_all_prob)
    aug_prob_nor = torch.max(aug_prob_nor, dim=0)[0]
    pseudo_uncert = (aug_prob_nor > thres) & (aug_prob_nor < 1 - thres)
    pseudo_uncert = pseudo_uncert.int().unsqueeze(0).long().to(device)
    pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().unsqueeze(0).long().to(device)
    prob_min = 0.3
    unct_no_aug_trans = no_aug_prob_nor.clone()
    unct_no_aug_trans[unct_no_aug_trans > 0.5] = 0
    unct_no_aug_trans[unct_no_aug_trans <= prob_min] = 0
    unct_no_aug_trans[unct_no_aug_trans > 0] = 1
    unct_no_aug_prob = unct_no_aug_trans.argmax(0)
    if and_fn:
        pseudo_uncert = unct_no_aug_prob.int() & pseudo_uncert.int()
    else:
        pseudo_uncert = pseudo_uncert.int()
    aug_avg_ent = sum(aug_all_ent) / len(aug_all_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0
    aug_std = torch.sigmoid(aug_output).std(dim=0, keepdim=True)
    if uncertain_to_positive:
        pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device) | pseudo_uncert
    else:
        pseudo_label = torch.tensor(no_aug_pseudo_label).clone().detach().to(device)
    aug_avg_ent = aug_avg_ent.clone().detach().unsqueeze(0).float().to(device)
    return (pseudo_label.unsqueeze(0).long(), pseudo_uncert.unsqueeze(0).long(), aug_std.unsqueeze(0).float())
