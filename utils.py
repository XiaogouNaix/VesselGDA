import numpy as np
import torch
import json
import SimpleITK as sitk
import os
import os.path as osp
import pickle
import yaml
from collections import OrderedDict, deque
from datetime import datetime
import random
DEBUG = True

def set_all_seeds(seed: int=42, deterministic_cudnn: bool=True):
    """Set Python, NumPy and PyTorch random seeds and configure cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
        else:
            torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(f'Random seed: {seed}')
    print(f'Deterministic cuDNN: {deterministic_cudnn}')
    if torch.cuda.is_available():
        print(f'CUDA random seeds initialized')

def debug_printer(func):

    def wrapper(*args, **kwargs):
        if DEBUG:
            return func(*args, **kwargs)
        else:
            pass
    return wrapper

@debug_printer
def write_nii(tensor, path):
    tensor = tensor.to(dtype=torch.float32)
    tensor = tensor.cpu().detach().numpy()[0][0]
    tensor = sitk.GetImageFromArray(tensor)
    sitk.WriteImage(tensor, osp.join('debug', path))

def separate_to_batches(array, batch_size):
    total_num = array.shape[0]
    num_batch = total_num // batch_size
    batch_lst = []
    for num in range(0, num_batch):
        batch_lst.append(array[num * batch_size:num * batch_size + batch_size])
    if total_num % batch_size != 0:
        batch_lst.append(array[num_batch * batch_size:])
    return batch_lst

def define_fd(path):
    if not osp.exists(path):
        os.makedirs(path)
    return path

def load_file(file_path, type='pkl'):
    if type == 'pkl' or type == 'pt':
        with open(file_path, 'rb') as f:
            a = pickle.load(f)
    elif type == 'json':
        with open(file_path, 'r') as f:
            a = json.load(f)
    elif type == 'npy':
        a = np.load(file_path)
    elif type == 'npz':
        a = np.load(file_path)
    elif type == 'nii':
        a = sitk.ReadImage(file_path)
        a = sitk.GetArrayFromImage(a)
    return a

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def write_yaml(yaml_path, dic):
    with open(yaml_path, 'w') as stream:
        try:
            yaml.dump(dic, stream)
        except yaml.YAMLError as exc:
            print(exc)

class TxtLogger:

    def __init__(self, txtPath, verbose=True):
        self.txtPath = txtPath
        self.verbose = verbose

    def write(self, strLine):
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        strLine = '{}: {}'.format(current_time_str, strLine)
        with open(self.txtPath, 'a') as f:
            f.write(strLine)
            f.write('\n')
        if self.verbose:
            print(strLine)

    def write_dict(self, dic):
        for key, value in dic.items():
            self.write('{}: {}'.format(key, value))

def dic_to_string(dic):
    string = ''
    for key, value in dic.items():
        if type(value) == float:
            string += f'{key}: {value:.6f}\n'
        else:
            string += f'{key}: {value}\n'
    return string

def safe_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1

def dict_element_add(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key] += value
        else:
            dic1[key] = value
    return dic1

def dict_element_lst_append(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            if type(dic1[key]) not in [list, tuple]:
                dic1[key] = [dic1[key]]
            dic1[key].append(value)
        else:
            dic1[key] = [value]
    return dic1

def dict_element_deque_append(dic1, dic2, maxlen):
    for key, value in dic2.items():
        if key in dic1:
            if type(dic1[key]) is not deque:
                raise TypeError(f'Expected deque for key {key}, but got {type(dic1[key])}')
            dic1[key].append(value)
        else:
            dic1[key] = deque([value], maxlen=maxlen)
    return dic1

def dict_element_divide(dic, divisor, smooth=1e-06):
    dic_0 = dic.copy()
    for key, value in dic_0.items():
        dic_0[key] = value / (divisor + smooth)
    return dic_0

def dict_stack(dic, axis=0):
    dic_0 = dic.copy()
    for key, value in dic_0.items():
        dic_0[key] = torch.stack(value, axis)
    return dic_0

def dict_to_device(dic, device):
    dic_0 = dic.copy()
    for key, value in dic_0.items():
        dic_0[key] = value.to(device=device, dtype=torch.float32)
    return dic_0

def insert_array(big_array, small_array, start_point):
    """
    Inserts a small numpy array into a larger array at a given starting point.
    If the insertion goes out of bounds, the small array is sliced to fit.

    Parameters:
        big_array (np.ndarray): The larger array.
        small_array (np.ndarray): The smaller array to be inserted.
        start_point (tuple): The starting index for insertion.

    Returns:
        np.ndarray: The modified larger array.
    """
    assert np.all(np.array(start_point) >= 0), 'Start point must be non-negative.'
    assert np.all(np.array(start_point) < big_array.shape), 'Start point must be within the bounds of the big array.'
    big_shape = big_array.shape
    small_shape = small_array.shape
    end_point = [min(start + size, big_shape[i]) for i, (start, size) in enumerate(zip(start_point, small_shape))]
    small_slices = tuple((slice(0, end - start) for start, end in zip(start_point, end_point)))
    big_slices = tuple((slice(start, end) for start, end in zip(start_point, end_point)))
    big_array[big_slices] = small_array[small_slices]
    return big_array

def safe_slice_array(array, start, end):
    """
    Safely slices an array given the start and end points.
    If the slice goes out of bounds, the slice is truncated to fit.

    Parameters:
        array (np.ndarray): The array to be sliced.
        start (tuple): The starting index for the slice.
        end (tuple): The ending index for the slice.

    Returns:
        np.ndarray: The sliced array.
    """
    assert len(start) == len(end), 'Start and end points must have the same number of dimensions.'
    assert len(start) == array.ndim, 'Start and end points must have the same number of dimensions as the array.'
    for idx in range(len(start)):
        assert start[idx] < end[idx], 'Start point must be less than end point in all dimensions.'
        if start[idx] < 0:
            start[idx] = 0
        if end[idx] > array.shape[idx]:
            end[idx] = array.shape[idx]
    arr_sliced = array[tuple((slice(start[i], end[i]) for i in range(len(start))))]
    return (arr_sliced, start, end)

def check_double_sigmoid(tensor):
    if 0 < tensor.min() and tensor.max() < 1:
        Warning('Tensor values are all in (0, 1), check if sigmoid has been applied twice.')

def safe_mean_torch(tensor, select=None):
    if select is not None:
        m = tensor[select].mean()
    else:
        m = tensor.mean()
    if torch.isnan(m):
        return torch.tensor(0.0, requires_grad=True)
    else:
        return m

def set_topk_values(tensor, k=5, largest=False, value=torch.inf):
    """
    For each row of the input tensor, set all values except the k smallest values to torch.inf.
    The positions of the k smallest values remain unchanged.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape [N1, N2].
        k (int): The number of smallest values to keep in each row. Default is 5.
    
    Returns:
        torch.Tensor: The modified tensor with values other than the smallest k set to torch.inf.
    """
    values, indices = torch.topk(tensor, k, dim=1, largest=largest, sorted=False)
    result = torch.full_like(tensor, value)
    result.scatter_(1, indices, values)
    return result

def get_center_value_3d(tensor):
    """
    Get the center value of a tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
    
    Returns:
        torch.Tensor: The center value of the tensor.
    """
    center = tensor[..., tensor.shape[0] // 2, tensor.shape[1] // 2, tensor.shape[2] // 2]
    return center

def select_voxels_by_mask_3d(tensor, mask):
    assert tensor.shape[-3:] == mask.shape, 'Tensor and mask must have the same spatial dimensions.'
    assert mask.type() in ['torch.BoolTensor', 'torch.cuda.BoolTensor'], 'Mask must be a boolean tensor.'
    idxs = torch.nonzero(mask, as_tuple=True)
    selected_voxels = tensor[..., idxs[0], idxs[1], idxs[2]]
    return selected_voxels

def EMA_update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    student_model_dict = model_student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = student_model_dict[key] * (1 - keep_rate) + value * keep_rate
        else:
            raise Exception('{} is not found in student model'.format(key))
    return new_teacher_dict

class PredictionStorage:

    def __init__(self, max_epoch=5):
        self.max_epoch = max_epoch
        self.storage = {}
        self.gt_storage = {}

    def add_predictions(self, case_id, predictions):
        self.storage = dict_element_deque_append(self.storage, {case_id: predictions}, maxlen=self.max_epoch)

    def initialize_gt(self, case_id, gt):
        if case_id not in self.gt_storage:
            self.gt_storage[case_id] = gt

    def get_gt(self, case_id):
        assert case_id in self.gt_storage, f'Ground truth for case_id {case_id} not initialized.'
        return self.gt_storage[case_id]

    def update_gt(self, case_id, gt):
        assert case_id in self.gt_storage, f'Ground truth for case_id {case_id} not initialized.'
        self.gt_storage[case_id] = gt

    def update_gt_ema(self, case_id, new_gt, keep_rate=0.9):
        assert case_id in self.gt_storage, f'Ground truth for case_id {case_id} not initialized.'
        old_gt = self.gt_storage[case_id]
        updated_gt = old_gt * keep_rate + new_gt * (1 - keep_rate)
        self.gt_storage[case_id] = updated_gt
