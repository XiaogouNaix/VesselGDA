import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as st_transforms
from torchvision.transforms.functional import rotate, InterpolationMode
from torch.nn import functional as F
from scipy.ndimage import label

def threshold_torch(tensor, threshold):
    return (tensor > threshold).float()

def crop_or_pad_center(arr, final_shape, center=None):
    """
    Crop or pad a numpy array to the specified final shape from the specified center.

    Parameters:
    arr (numpy.ndarray): The input array to be cropped or padded.
    final_shape (tuple): The desired shape of the output array.
    center (tuple, optional): The center point for cropping or padding. If None, the array's center is used.

    Returns:
    numpy.ndarray: The cropped or padded array.
    """
    input_shape = arr.shape
    if center is None:
        center = tuple((dim // 2 for dim in input_shape))
    result = np.zeros(final_shape, dtype=arr.dtype)
    input_slices = []
    result_slices = []
    for dim in range(len(final_shape)):
        input_len = input_shape[dim]
        final_len = final_shape[dim]
        center_dim = center[dim]
        start_in = max(center_dim - final_len // 2, 0)
        end_in = min(center_dim + final_len // 2, input_len)
        start_out = max(final_len // 2 - center_dim, 0)
        end_out = start_out + (end_in - start_in)
        input_slices.append(slice(start_in, end_in))
        result_slices.append(slice(start_out, end_out))
    result[tuple(result_slices)] = arr[tuple(input_slices)]
    return result

def choose_max_connected_component(arr):
    labels = measure.label(arr, connectivity=1)
    props = measure.regionprops(labels)
    max_area = 0
    max_label = 0
    for prop in props:
        if prop.area > max_area:
            max_area = prop.area
            max_label = prop.label
    return labels == max_label

def rotate_image_cv2(image, angle, background_color=(0, 0, 0)):
    """Rotate an image around its center using OpenCV."""
    image = image.transpose(1, 2, 0)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_w = w
    new_h = h
    if background_color == 'min':
        min_val = np.min(image)
        background_color = np.float64(min_val)
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=background_color)
    if image.shape[2] == 1:
        rotated = rotated[:, :, np.newaxis]
    image = rotated.transpose(2, 0, 1)
    return image

def flip_3D_torch(cube, axis):
    assert len(cube.shape) == 5
    if axis == 0:
        return cube
    elif axis == 1:
        return cube.flip(2)
    elif axis == 2:
        return cube.flip(3)
    elif axis == 3:
        return cube.flip(4)
    elif axis == 4:
        cube = cube.flip(2)
        return cube.flip(3)
    elif axis == 5:
        cube = cube.flip(2)
        return cube.flip(4)
    elif axis == 6:
        cube = cube.flip(3)
        return cube.flip(4)
    elif axis == 7:
        cube = cube.flip(2)
        cube = cube.flip(3)
        return cube.flip(4)
    else:
        raise ValueError('axis should be in [0, 1, 2, 3, 4, 5, 6, 7]')

def rotate_and_flip_2D_torch(image, code, background_color=None):
    assert len(image.shape) == 4
    if code == 0:
        return image
    elif code in [1, 2, 3]:
        return torch.rot90(image, k=code, dims=(2, 3))
    elif code == 4:
        return torch.flip(image, dims=(3,))
    elif code == 5:
        return torch.flip(image, dims=(2,))
    elif code == 6:
        return torch.flip(image, dims=(2, 3))

def rotate_2D_torch(image, code, background_color=None):
    assert len(image.shape) == 4
    if background_color is None:
        if image.shape[1] == 3 and image.dtype == torch.uint8:
            background_color = (0, 0, 0) if background_color is None else background_color
        else:
            background_color = 'min'
    degree = code * 45
    rotated_images = []
    for i in range(image.shape[0]):
        img = image[i]
        rotated_img = rotate(img, degree, interpolation=InterpolationMode.BILINEAR, fill=torch.min(img).item() if background_color == 'min' else background_color)
        rotated_images.append(rotated_img)
    rotated_images = torch.stack(rotated_images, dim=0)
    return rotated_images

class StrongAugmentation(nn.Module):

    def __init__(self):
        super(StrongAugmentation, self).__init__()
        self.augmentation = []
        self.augmentation.append(st_transforms.RandomApply([st_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6))
        self.augmentation.append(st_transforms.RandomGrayscale(p=0.2))
        self.strong_aug = st_transforms.Compose(self.augmentation)

    def forward(self, img):
        s_input = self.strong_aug(img)
        return s_input

class RandomGaussianBlur(nn.Module):

    def __init__(self, kernel_size_range=(9, 21), sigma_range=(1.0, 5.0), downsample=2, p=0.5):
        super().__init__()
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p
        self.downsample = downsample

    def forward(self, img):
        orig_img = torch.clone(img)
        if torch.rand(1) > self.p:
            return img
        img = F.interpolate(img, scale_factor=1 / self.downsample, mode='bilinear', align_corners=False)
        kernel_size = torch.randint(self.kernel_size_range[0] // 2 * 2 + 1, self.kernel_size_range[1] // 2 * 2 + 2, (1,)).item()
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        sigma = torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]).item()
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        img_expanded = img
        kernel = kernel_2d.repeat(3, 3, 1, 1).to(img.device)
        padding = kernel_size // 2
        blurred = F.conv2d(img_expanded, kernel, padding=padding)
        blurred = F.interpolate(blurred, size=orig_img.shape[2:], mode='bilinear', align_corners=False)
        return blurred

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size_range={self.kernel_size_range}, sigma_range={self.sigma_range}, p={self.p})'

class RandomContrastStretch(nn.Module):

    def __init__(self, output_range_min=(-0.2, 0.2), output_range_max=(0.8, 1.2), p=0.5):
        super().__init__()
        self.output_range_min = output_range_min
        self.output_range_max = output_range_max
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        a = torch.empty(1).uniform_(self.output_range_min[0], self.output_range_min[1]).item()
        b = torch.empty(1).uniform_(self.output_range_max[0], self.output_range_max[1]).item()
        a, b = (min(a, b), max(a, b))
        c = img.min()
        d = img.max()
        if d - c < 1e-05:
            return img
        img_stretched = a + (img - c) * (b - a) / (d - c)
        return img_stretched

    def __repr__(self):
        return f'{self.__class__.__name__}(output_range_min={self.output_range_min}, output_range_max={self.output_range_max}, p={self.p})'
import numpy as np
from skimage import measure
import numpy as np
from scipy.ndimage import label

def get_intersecting_components(matrix1, matrix2):
    """Return the 8-connected components of matrix2 that intersect the foreground of matrix1."""
    labeled_array, num_features = label(matrix2, structure=np.ones((3, 3)))
    intersecting_labels = set(labeled_array[matrix1 > 0])
    if 0 in intersecting_labels:
        intersecting_labels.remove(0)
    mask = np.isin(labeled_array, list(intersecting_labels))
    return mask.astype(np.uint8)
