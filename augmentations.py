import numpy as np
import torch
import torch.nn as nn
import torchio as tio
import random
from torchvision.transforms import functional as TF
from baseline_dual_normalization.bezier_aug import nonlinear_transformation_torch

class MultiImageRandomTransform(torch.nn.Module):

    def __init__(self, rotation_range=(-180, 180), p_flip=0.5, p_90_rotate=0.5):
        """Initialize MultiImageRandomTransform."""
        super().__init__()
        self.rotation_range = rotation_range
        self.p_flip = p_flip
        self.p_90_rotate = p_90_rotate

    def forward(self, imgs_dic):
        """Apply MultiImageRandomTransform to the input tensors."""
        angle = self._get_random_angle()
        h_flip = random.random() < self.p_flip
        v_flip = random.random() < self.p_flip
        augmented_dic = {}
        for k, img in imgs_dic.items():
            img = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
            img = self._apply_transform(img, angle, h_flip, v_flip)
            augmented_dic[k] = img
        return augmented_dic

    def _get_random_angle(self):
        """Draw a random rotation angle from the configured range."""
        angle = 0
        if self.rotation_range is not None:
            angle += random.uniform(*self.rotation_range)
        if random.random() < self.p_90_rotate:
            angle += random.choice([0, 90, 180, 270])
        return angle

    def _apply_transform(self, img, angle, h_flip, v_flip):
        """Apply a shared spatial transform to the image and its auxiliary arrays."""
        img = TF.rotate(img, angle)
        if h_flip:
            img = TF.hflip(img)
        if v_flip:
            img = TF.vflip(img)
        return img

def random_translation_3d(arr_lst, max_translation=5):
    """
    Randomly translate a 3D array along each axis.
    Args:
        arr_lst (lst or dic): list of 3D array to be translated.
    Returns:
        np.ndarray: Translated 3D array.
    """
    translation = np.random.randint(-max_translation, max_translation, size=3)
    if type(arr_lst) in [list, tuple]:
        for i in range(len(arr_lst)):
            arr_lst[i] = np.roll(arr_lst[i], translation, axis=(0, 1, 2))
    elif type(arr_lst) == dict:
        for k, v in arr_lst.items():
            arr_lst[k] = np.roll(v, translation, axis=(0, 1, 2))
    else:
        raise ValueError('arr_lst should be either list, tuple or dict.')
    return arr_lst

def random_flipping_3d(arr_dic, axes=(1, 2)):
    """
    Randomly flip a 3D array along each axis.
    Args:
        arr_dic (dic): dictionary of 3D array to be flipped.
        axes (tuple): axes along which to flip.
    Returns:
        np.ndarray: Flipped 3D array.
    """
    do_flip_axes = []
    for axis in axes:
        if np.random.rand() > 0.5:
            do_flip_axes.append(axis)
    for k, v in arr_dic.items():
        for axis in do_flip_axes:
            arr_dic[k] = np.flip(v, axis=axis).copy()
    return arr_dic

class GIN(nn.Module):

    def __init__(self, n_channels, mid_channels, num_layers):
        super(GIN, self).__init__()
        self.n_channels = n_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers
        self.lst = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lst.append(nn.Conv3d(n_channels, mid_channels, kernel_size=3, padding=1))
            else:
                self.lst.append(nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1))
            self.lst.append(nn.ReLU(inplace=True))
        self.lst.append(nn.Conv3d(mid_channels, n_channels, kernel_size=3, padding=1))

    def _init_params(self):
        for i in range(self.num_layers):
            nn.init.normal_(self.lst[i * 2].weight)

    def forward(self, x):
        x_ori = x.clone()
        for i in range(self.num_layers):
            x = self.lst[i * 2](x)
            x = self.lst[i * 2 + 1](x)
        x = self.lst[-1](x)
        proportion = torch.rand(1).item()
        x_norm = torch.norm(x, p=2, dim=(1, 2, 3, 4), keepdim=True)
        x_ori_norm = torch.norm(x_ori, p=2, dim=(1, 2, 3, 4), keepdim=True)
        x = x / (x_norm + 1e-06) * (x_ori_norm + 1e-06)
        x = proportion * x + (1 - proportion) * x_ori
        return x

class Bezier(nn.Module):

    def __init__(self, flip=False, prob=0.5, num_free_coordinates=1, max_rnd_val=0.5, min_rnd_val=0, same_mean=True):
        super(Bezier, self).__init__()
        self.flip = flip
        self.prob = prob
        self.num_free_coordinates = num_free_coordinates
        self.max_rnd_val = max_rnd_val
        self.min_rnd_val = min_rnd_val
        self.same_mean = same_mean

    def forward(self, x):
        if len(x.shape) == 5:
            dims = (2, 3, 4)
        elif len(x.shape) == 4:
            dims = (2, 3)
        else:
            raise ValueError('Input x must be 4D or 5D tensor.')
        x_mean = torch.mean(x, dim=dims, keepdim=True)
        x_aug = nonlinear_transformation_torch(x, self.flip, self.prob, self.num_free_coordinates, self.max_rnd_val, self.min_rnd_val)
        if self.same_mean:
            x_aug = x_aug + (x_mean - torch.mean(x_aug, dim=dims, keepdim=True))
        return x_aug

class RandAug(nn.Module):

    def __init__(self, n_augs):
        super(RandAug, self).__init__()
        augs_lst = [tio.transforms.RandomGhosting(), tio.transforms.RandomBiasField(), tio.transforms.RandomBlur(), tio.transforms.RandomElasticDeformation(), nn.Identity()]
        self.augs = random.choices(augs_lst, k=n_augs)

    def forward(self, x):
        device = x.device
        squeeze_flag = 0
        if len(x.shape) == 4:
            squeeze_flag = 1
            x = x.unsqueeze(2)
        x = x.cpu().numpy()
        for i in range(len(x)):
            for aug in self.augs:
                x[i] = aug(x[i])
        x = torch.from_numpy(x).to(device)
        if squeeze_flag:
            x = x.squeeze(2)
        return x

def shuffle_remap(data, ranges=[-1, 1], rand_point=[2, 50]):
    control_point = random.randint(rand_point[0], rand_point[1])
    distribu = torch.rand(control_point) * (ranges[1] - ranges[0]) + ranges[0]
    distribu, _ = torch.sort(distribu)
    distribu = torch.cat([torch.tensor([ranges[0]]), distribu])
    distribu = torch.cat([distribu, torch.tensor([ranges[1]])])
    shuffle_part = torch.randperm(control_point + 1)
    new_image = torch.zeros_like(data)
    for i in range(control_point + 1):
        target_part = shuffle_part[i]
        min1, max1 = (distribu[i], distribu[i + 1])
        min2, max2 = (distribu[target_part], distribu[target_part + 1])
        coord = torch.where((min1 <= data) & (data < max1))
        new_image[coord] = (data[coord] - min1) / (max1 - min1) * (max2 - min2) + min2
    new_image = torch.clamp(new_image, ranges[0], ranges[1])
    return new_image

class ShuffleRemap(nn.Module):

    def __init__(self, rand_point_lo=2, rand_point_hi=50, same_mean=True):
        super(ShuffleRemap, self).__init__()
        self.rand_point = [rand_point_lo, rand_point_hi]
        self.same_mean = same_mean

    def forward(self, x):
        min_val = torch.min(x).item()
        max_val = torch.max(x).item()
        mean = torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)
        device = x.device
        x = x.cpu()
        x = shuffle_remap(x, [min_val, max_val], self.rand_point)
        x = x.to(device)
        if self.same_mean:
            x = x + (mean - torch.mean(x, dim=(1, 2, 3, 4), keepdim=True))
        return x
