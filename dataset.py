import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import os.path as osp
import SimpleITK as sitk
import cv2
from skimage.transform import resize
from utils_preprocess import crop_or_pad_center
from utils import load_file
import augmentations as aug

def znorm(image):
    image = (image - np.mean(image)) / np.std(image)
    return image

def znorm1(image):
    mask = (image == image.min()) | (image == image.max())
    image = (image - np.mean(image[~mask])) / np.std(image[~mask]) * 1.5
    return image

def minmax(image, target_min=0, target_max=1):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

class CustomDataset(Dataset):

    def __init__(self, base_data_path, mode='train', test_fd_str='Ts', image_format='.nii.gz', label_format='.nii.gz', last_stage_mask_path=None, invert_grayscale=False, min_size=None, wanted_size=None, length_shorten=None, allow_no_label=False, only_no_label=False, norm_method={'name': 'znorm'}, augs={'translation': False, 'flipping': False}, other_aux=[], load_mask_pseudo=None, image_fd_str=None, wanted_names=None):
        assert mode in ['train', 'val', 'test']
        self.base_data_path = base_data_path
        self.invert_grayscale = invert_grayscale
        self.min_size = min_size
        self.wanted_size = wanted_size
        self.image_format = image_format
        self.label_format = label_format
        self.allow_no_label = allow_no_label
        self.augs = augs
        self.norm_method = norm_method
        self.load_mask_pseudo = load_mask_pseudo
        image_fd = 'imagesTr' if mode in ['train', 'val'] else 'images' + 'Tr' if test_fd_str != 'Ts' else 'images' + test_fd_str
        if image_fd_str is not None:
            image_fd = 'images' + image_fd_str
        label_fd = 'labelsTr' if mode in ['train', 'val'] else 'labels' + test_fd_str
        if only_no_label:
            self.names = [name for name in os.listdir(osp.join(base_data_path, image_fd)) if not osp.exists(osp.join(base_data_path, label_fd, name))]
        else:
            self.names = os.listdir(osp.join(base_data_path, label_fd if not allow_no_label else image_fd))
        self.image_path_base = osp.join(base_data_path, image_fd)
        self.label_path_base = osp.join(base_data_path, label_fd)
        self.last_stage_mask_path = last_stage_mask_path
        self.other_aux = other_aux
        self.names.sort()
        l = len(self.names)
        if wanted_names is not None:
            wanted_names_lst = pickle.load(open(wanted_names[0], 'rb'))[wanted_names[1]]
            self.names = [name for name in self.names if name.rstrip(image_format) in wanted_names_lst]
        val_ratio = 0.1
        if mode == 'train':
            self.names = self.names[:int(l * (1 - val_ratio))]
        elif mode == 'val':
            self.names = self.names[int(l * (1 - val_ratio)):]
        if length_shorten is not None:
            if type(length_shorten) == int:
                self.names = self.names[:length_shorten]
            elif type(length_shorten) == tuple and len(length_shorten) == 2:
                self.names = self.names[length_shorten[0]:length_shorten[1]]
            print(f'Dataset length shortened to: {len(self.names)}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label_path = osp.join(self.label_path_base, name)
        name_base = name[:-len(self.label_format)]
        image_name = name
        image_path = osp.join(self.image_path_base, image_name)
        image = sitk.ReadImage(image_path)
        spacing = image.GetSpacing()
        image_arr = sitk.GetArrayFromImage(image)
        if osp.exists(label_path):
            label = sitk.ReadImage(label_path)
            label_arr = sitk.GetArrayFromImage(label)
        elif self.allow_no_label:
            label_arr = np.zeros_like(image_arr)
        else:
            raise FileNotFoundError(f'Label file not found: {label_path}')
        other_arr = {}
        if self.last_stage_mask_path is not None:
            for k, v in self.last_stage_mask_path.items():
                other_path = osp.join(v, name)
                if osp.exists(other_path):
                    other = sitk.ReadImage(other_path)
                    other_arr[k] = sitk.GetArrayFromImage(other)
                else:
                    other_arr[k] = np.array([np.nan])
            if 'pseudo' in self.last_stage_mask_path.keys() and 'mask' in self.last_stage_mask_path.keys():
                pseudo = other_arr['pseudo']
                mask = other_arr['mask']
                area_interest = (pseudo > 0) | (1 - mask > 0)
                area_interest = area_interest.astype(np.float32)
                other_arr['area_interest'] = area_interest
        if self.load_mask_pseudo is not None:
            mask_path = osp.join(self.load_mask_pseudo, name_base + '_mask.pkl')
            pseudo_path = osp.join(self.load_mask_pseudo, name_base + '_pseudo.pkl')
            mask = pickle.load(open(mask_path, 'rb'))
            pseudo = pickle.load(open(pseudo_path, 'rb'))
            other_arr['uncertainty_mask'] = mask
            other_arr['pseudo_label'] = pseudo
        other_aux_dic = {}
        for dic0 in self.other_aux:
            other_path = osp.join(dic0['aux_path'], name_base + '.' + dic0['type'])
            if osp.exists(other_path):
                other = load_file(other_path, dic0['type'])
                other_aux_dic[dic0['name']] = other
            else:
                other_aux_dic[dic0['name']] = None
        if self.invert_grayscale:
            image_arr = image_arr.max() - image_arr
        if self.norm_method['name'] == 'znorm':
            image_arr = znorm(image_arr)
        elif self.norm_method['name'] == 'znorm_scale':
            assert len(self.norm_method['params']) == 1
            image_arr = znorm(image_arr)
            image_arr = image_arr * self.norm_method['params'][0]
        elif self.norm_method['name'] == 'znorm1':
            image_arr = znorm1(image_arr)
        elif self.norm_method['name'] == 'znorm1_scale':
            assert len(self.norm_method['params']) == 1
            image_arr = znorm1(image_arr)
            image_arr = image_arr * self.norm_method['params'][0]
        elif self.norm_method['name'] == 'minmax':
            assert len(self.norm_method['params']) == 2
            image_arr = minmax(image_arr, self.norm_method['params'][0], self.norm_method['params'][1])
        elif self.norm_method['name'] == 'clip':
            assert len(self.norm_method['params']) == 2
            image_arr = np.clip(image_arr, self.norm_method['params'][0], self.norm_method['params'][1])
        elif self.norm_method['name'] == 'clip_znorm':
            assert len(self.norm_method['params']) == 2
            image_arr = np.clip(image_arr, self.norm_method['params'][0], self.norm_method['params'][1])
            image_arr = znorm1(image_arr)
        elif self.norm_method['name'] == 'znorm_clip':
            assert len(self.norm_method['params']) == 2
            image_arr = znorm(image_arr)
            image_arr = np.clip(image_arr, self.norm_method['params'][0], self.norm_method['params'][1])
        elif self.norm_method['name'] == 'clip_minmax':
            assert len(self.norm_method['params']) == 4
            image_arr = np.clip(image_arr, self.norm_method['params'][0], self.norm_method['params'][1])
            image_arr = minmax(image_arr, self.norm_method['params'][2], self.norm_method['params'][3])
        elif self.norm_method['name'] == 'binarize':
            assert len(self.norm_method['params']) == 1
            image_arr = (image_arr > self.norm_method['params'][0]).astype(np.float32)
        elif self.norm_method['name'] == 'none':
            pass
        else:
            raise NotImplementedError(f"Normalization method {self.norm_method['name']} not implemented.")
        if self.wanted_size is not None:
            image_arr = crop_or_pad_center(image_arr, self.wanted_size)
            label_arr = crop_or_pad_center(label_arr, self.wanted_size)
            if self.last_stage_mask_path is not None:
                for k, v in other_arr.items():
                    other_arr[k] = crop_or_pad_center(v, self.wanted_size)
        elif self.min_size is not None:
            if image_arr.shape[0] < self.min_size[0]:
                pad = self.min_size[0] - image_arr.shape[0]
                image_arr = np.pad(image_arr, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode='constant')
                label_arr = np.pad(label_arr, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode='constant')
        if self.augs['translation']:
            image_arr, label_arr, other_arr = self._do_aug(image_arr, label_arr, aug.random_translation_3d, other_arr)
        if self.augs['flipping']:
            image_arr, label_arr, other_arr = self._do_aug(image_arr, label_arr, aug.random_flipping_3d, other_arr)
        image_arr = np.expand_dims(image_arr, axis=0)
        label_arr = np.expand_dims(label_arr, axis=0)
        if self.last_stage_mask_path is not None:
            for k, v in other_arr.items():
                other_arr[k] = np.expand_dims(other_arr[k], axis=0)
        dic = {'image': image_arr, 'label': label_arr, 'spacing': spacing, 'name': name_base}
        if other_arr:
            dic.update(other_arr)
        dic.update(other_aux_dic)
        return dic

    def _do_aug(self, image_arr, label_arr, aug_func, other_arr):
        if self.last_stage_mask_path is not None:
            assert other_arr is not None
        dic = {'image': image_arr, 'label': label_arr}
        dic.update(other_arr)
        lst = aug_func(dic)
        image_arr, label_arr = (lst['image'], lst['label'])
        for k, v in other_arr.items():
            other_arr[k] = lst[k]
        return (image_arr, label_arr, other_arr)

class Dataset2d(Dataset):

    def __init__(self, base_data_path, mode='train', test_fd_str='Ts', image_format='.png', label_format='.png', last_stage_mask_path=None, invert_grayscale=False, min_size=None, wanted_size=None, length_shorten=None, allow_no_label=False, only_no_label=False, norm_method={'name': 'znorm'}, augs={'translation': False, 'flipping': False}, other_aux=[], load_mask_pseudo=None, image_fd_str=None, wanted_names=None, convert_bgr_to_rgb=True):
        assert mode in ['train', 'test']
        self.mode = mode
        self.base_data_path = base_data_path
        self.invert_grayscale = invert_grayscale
        self.min_size = min_size
        self.wanted_size = wanted_size
        self.image_format = image_format
        self.label_format = label_format
        self.allow_no_label = allow_no_label
        self.only_no_label = only_no_label
        self.augs = augs
        self.norm_method = norm_method
        self.load_mask_pseudo = load_mask_pseudo
        self.convert_bgr_to_rgb = convert_bgr_to_rgb
        self.names = [name for name in os.listdir(base_data_path) if osp.isdir(osp.join(base_data_path, name))]
        self.image_path_base = base_data_path
        self.last_stage_mask_path = last_stage_mask_path
        self.other_aux = other_aux
        self.names.sort()
        if wanted_names is not None:
            wanted_names_lst = pickle.load(open(wanted_names[0], 'rb'))[wanted_names[1]]
            self.names = [name for name in self.names if name.rstrip(image_format) in wanted_names_lst]
        if length_shorten is not None:
            if type(length_shorten) == int:
                self.names = self.names[:length_shorten]
            elif type(length_shorten) in [tuple, list] and len(length_shorten) == 2:
                self.names = self.names[length_shorten[0]:length_shorten[1]]

    def __getitem__(self, idx):
        name = self.names[idx]
        label_path = osp.join(self.image_path_base, name, 'label.png')
        image_path = osp.join(self.image_path_base, name, 'image.png')
        image_arr = cv2.imread(image_path)
        if self.convert_bgr_to_rgb:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        if osp.exists(label_path) and not self.only_no_label:
            label_arr = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        elif self.allow_no_label:
            label_arr = np.zeros_like(image_arr[:, :, 0])
        else:
            raise FileNotFoundError(f'Label file not found: {label_path}')
        other_arr = {}
        if self.last_stage_mask_path is not None:
            for k, v in self.last_stage_mask_path.items():
                other_path = osp.join(v, name + '.pt')
                if osp.exists(other_path):
                    other_arr[k] = pickle.load(open(other_path, 'rb'))['arr']
                    if len(other_arr[k].shape) == 2:
                        other_arr[k] = np.expand_dims(other_arr[k], axis=0)
                else:
                    other_arr[k] = None
        if self.load_mask_pseudo is not None:
            mask_path = osp.join(self.load_mask_pseudo, name + '_mask.pkl')
            pseudo_path = osp.join(self.load_mask_pseudo, name + '_pseudo.pkl')
            mask = pickle.load(open(mask_path, 'rb'))
            pseudo = pickle.load(open(pseudo_path, 'rb'))
            other_arr['uncertainty_mask'] = mask[0].numpy()
            other_arr['pseudo_label'] = pseudo[0].numpy()
        other_aux_dic = {}
        for dic0 in self.other_aux:
            other_path = osp.join(dic0['aux_path'], name + '.' + dic0['type'])
            if osp.exists(other_path):
                other = load_file(other_path, dic0['type'])
                other_aux_dic[dic0['name']] = other
            else:
                other_aux_dic[dic0['name']] = None
        if self.invert_grayscale:
            image_arr = image_arr.max() - image_arr
        if self.norm_method['name'] == 'znorm':
            image_arr = znorm(image_arr)
        elif self.norm_method['name'] == 'png_norm':
            image_arr = image_arr / 255.0
        elif self.norm_method['name'] == 'none':
            pass
        else:
            raise NotImplementedError(f"Normalization method {self.norm_method['name']} not implemented.")
        if self.wanted_size is not None:
            image_arr = cv2.resize(image_arr, self.wanted_size, interpolation=cv2.INTER_LINEAR)
            label_arr = cv2.resize(label_arr, self.wanted_size)
            if self.last_stage_mask_path is not None:
                for k, v in other_arr.items():
                    if v is not None:
                        other_arr[k] = resize(v, (v.shape[0], *self.wanted_size))
        label_arr = np.expand_dims(label_arr, axis=0)
        image_arr = np.transpose(image_arr, (2, 0, 1))
        if self.augs is not None:
            rotation_range = None
            p_flip = 0
            p_90_rotate = 0
            if self.augs.get('rotate_random'):
                rotation_range = (-90, 90)
            if self.augs.get('rotate_90'):
                p_90_rotate = 1
            if self.augs.get('flip'):
                p_flip = 0.5
            aug_func = aug.MultiImageRandomTransform(rotation_range=rotation_range, p_flip=p_flip, p_90_rotate=p_90_rotate)
            image_arr, label_arr, other_arr = self._do_aug(image_arr, label_arr, aug_func, other_arr)
        label_arr = label_arr / 255.0
        label_arr[label_arr > 0.5] = 1.0
        label_arr[label_arr <= 0.5] = 0.0
        dic = {'image': image_arr, 'label': label_arr, 'name': name}
        if self.last_stage_mask_path is not None:
            dic.update(other_arr)
        dic.update(other_aux_dic)
        return dic

    def __len__(self):
        return len(self.names)

    def _do_aug(self, image_arr, label_arr, aug_func, other_arr):
        if self.last_stage_mask_path is not None:
            assert other_arr is not None
        dic = {'image': image_arr, 'label': label_arr}
        dic.update(other_arr)
        lst = aug_func(dic)
        image_arr, label_arr = (lst['image'], lst['label'])
        for k, v in other_arr.items():
            other_arr[k] = lst[k]
        return (image_arr, label_arr, other_arr)
