from scipy.ndimage import distance_transform_edt
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Sampler
from typing import List, Tuple, Optional, Union, Dict
from scipy.ndimage import label

def distance_transform_edt_torch(mask, sampling=None):
    device = mask.device
    mask = mask.cpu().numpy()
    sh = mask.shape
    mask = mask.squeeze(1)
    dist = np.zeros_like(mask)
    for i in range(sh[0]):
        dist[i] = distance_transform_edt(mask[i], sampling=sampling) - distance_transform_edt(1 - mask[i], sampling=sampling)
    dist = torch.from_numpy(dist).unsqueeze(1)
    return dist.to(device)

def component_weighted_distance_transform_torch(large_mask: torch.Tensor, area_scale: float=5000.0, min_weight: float=0.05, top_k: int=15):
    """
    Component-weighted distance transform (top-k CC only)

    Args:
        large_mask: (B, 1, H, W) or (B, 1, D, H, W), binary {0,1}
        area_scale: controls how fast weight saturates
        min_weight: lower bound to avoid infinite distance
        top_k: only use largest k connected components

    Returns:
        dist_map: same shape as large_mask, float
    """
    device = large_mask.device
    mask_np = large_mask.cpu().numpy()
    B = mask_np.shape[0]
    out_dist = []
    for b in range(B):
        m = mask_np[b, 0] > 0
        cc_map, num_cc = label(m)
        if num_cc == 0:
            dist = distance_transform_edt(~m)
            out_dist.append(dist)
            continue
        cc_areas = []
        for cc_id in range(1, num_cc + 1):
            area = np.sum(cc_map == cc_id)
            cc_areas.append((cc_id, area))
        cc_areas.sort(key=lambda x: x[1], reverse=True)
        selected_ccs = cc_areas[:top_k]
        dist_maps = []
        for cc_id, area in selected_ccs:
            w = min(1.0, area / area_scale)
            w = max(w, min_weight)
            cc_mask = cc_map == cc_id
            dist_cc = distance_transform_edt(~cc_mask)
            dist_maps.append(dist_cc / w)
        dist_final = np.minimum.reduce(dist_maps)
        out_dist.append(dist_final)
    out_dist = np.stack(out_dist, axis=0)
    out_dist = torch.from_numpy(out_dist).unsqueeze(1).float().to(device)
    return out_dist

class GPULabelSampler(Sampler):
    """Sample volumetric patches on the tensor device using class-dependent probabilities."""

    def __init__(self, patch_size: Union[Tuple[int, int], Tuple[int, int, int]], samples_per_volume: int=16, prob_map: str='weighted', prob_weights: Optional[Dict[int, float]]=None, padding_mode: str='constant', max_iter: int=1000):
        super().__init__()
        self.patch_size = torch.tensor(patch_size, dtype=torch.long)
        self.samples_per_volume = samples_per_volume
        self.prob_map = prob_map
        self.prob_weights = prob_weights
        self.padding_mode = padding_mode
        self.max_iter = max_iter
        self.padding = [size // 2 for size in patch_size for _ in range(2)]
        if len(patch_size) in [2, 3]:
            self.padding = self.padding[::-1]

    def __call__(self, volumes: Dict[str, torch.Tensor], label_map: torch.Tensor, probability_map: Optional[torch.Tensor]=None) -> Dict[str, List[torch.Tensor]]:
        """Sample aligned patches from the supplied volumes."""
        self._validate_volumes(volumes, label_map)
        if probability_map is None:
            probability_map = self._create_probability_map(label_map)
        else:
            probability_map = probability_map
        return self._sample_patches_gpu(volumes, label_map, probability_map)

    def _validate_volumes(self, volumes: Dict[str, torch.Tensor], label_map: torch.Tensor):
        """Validate the dimensions and spatial agreement of the supplied volumes."""
        if not volumes:
            raise ValueError('volumes must not be empty')
        first_volume = next(iter(volumes.values()))
        ref_spatial_dims = first_volume.shape[1:]
        if label_map.shape[1:] != ref_spatial_dims:
            raise ValueError(f'label_map spatial shape {label_map.shape[1:]} differs from reference {ref_spatial_dims}')
        for name, volume in volumes.items():
            if volume.shape[1:] != ref_spatial_dims:
                raise ValueError(f"Volume '{name}' spatial shape {volume.shape[1:]} differs from reference {ref_spatial_dims}")
            if volume.device != label_map.device:
                raise ValueError(f"Volume '{name}' and label_map must be on the same device")

    def _create_probability_map(self, label_map: torch.Tensor) -> torch.Tensor:
        """Construct the sampling probability map from class labels and class weights."""
        if self.prob_map == 'binary':
            prob_map = (label_map > 0).float()
        elif self.prob_map == 'multiclass':
            unique_labels = torch.unique(label_map)
            prob_map = torch.zeros_like(label_map, dtype=torch.float)
            for label in unique_labels:
                if label > 0:
                    mask = label_map == label
                    prob_map[mask] = 1.0 / len(unique_labels[unique_labels > 0])
        elif self.prob_map == 'weighted':
            unique_labels, counts = torch.unique(label_map, return_counts=True)
            prob_map = torch.zeros_like(label_map, dtype=torch.float)
            total_pixels = label_map.numel()
            for label, count in zip(unique_labels, counts):
                if label > 0 and count > 0:
                    weight = total_pixels / (count * len(unique_labels[unique_labels > 0]))
                    prob_map[label_map == label] = weight
        elif self.prob_map == 'weighted_custom':
            if self.prob_weights is None:
                raise ValueError("prob_weights is required for 'weighted_custom' mode")
            prob_map = torch.zeros_like(label_map, dtype=torch.float)
            for label, weight in self.prob_weights.items():
                prob_map[label_map == label] = weight
        else:
            raise ValueError(f'Unsupported prob_map mode: {self.prob_map}')
        if prob_map.sum() > 0:
            prob_map = prob_map / prob_map.sum()
        else:
            prob_map = torch.ones_like(prob_map) / prob_map.numel()
        return prob_map

    def _sample_patches_gpu(self, volumes: Dict[str, torch.Tensor], label_map: torch.Tensor, probability_map: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """Sample patches on the tensor device."""
        patches_dict = {name: [] for name in volumes.keys()}
        spatial_dims = len(next(iter(volumes.values())).shape) - 1
        first_volume = next(iter(volumes.values()))
        original_spatial_shape = first_volume.shape[1:]
        padded_volumes = {}
        for name, volume in volumes.items():
            if spatial_dims == 2:
                padded_volumes[name] = F.pad(volume.unsqueeze(0), self.padding, mode=self.padding_mode).squeeze(0)
            else:
                padded_volumes[name] = F.pad(volume, self.padding, mode=self.padding_mode)
        if spatial_dims == 2:
            padded_label = F.pad(label_map.unsqueeze(0), self.padding, mode=self.padding_mode).squeeze(0)
        else:
            padded_label = F.pad(label_map, self.padding, mode=self.padding_mode)
        if spatial_dims == 2:
            pad_start = torch.tensor([self.padding[1], self.padding[3]], device=first_volume.device, dtype=torch.long)
        else:
            pad_start = torch.tensor([self.padding[4], self.padding[2], self.padding[0]], device=first_volume.device, dtype=torch.long)
        flat_prob = probability_map.flatten()
        for _ in range(self.samples_per_volume):
            patches = self._sample_single_patch(padded_volumes, padded_label, flat_prob, spatial_dims, original_spatial_shape, pad_start)
            if patches is not None:
                for name, patch in patches.items():
                    patches_dict[name].append(patch)
        return patches_dict

    def _sample_single_patch(self, padded_volumes: Dict[str, torch.Tensor], padded_label: torch.Tensor, flat_prob: torch.Tensor, spatial_dims: int, original_spatial_shape: Tuple[int, ...], pad_start: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Sample one valid patch according to the probability map."""
        for attempt in range(self.max_iter):
            center_idx = torch.multinomial(flat_prob, 1, replacement=True)
            center_coord_original = torch.zeros(spatial_dims, dtype=torch.long, device=flat_prob.device)
            remainder = center_idx.item()
            for dim in range(spatial_dims - 1, -1, -1):
                dim_size = original_spatial_shape[dim]
                center_coord_original[dim] = remainder % dim_size
                remainder = remainder // dim_size
            center_coord = center_coord_original + pad_start
            self.patch_size = self.patch_size.to(center_coord.device)
            half_patch = self.patch_size // 2
            start_coord = center_coord - half_patch
            end_coord = start_coord + self.patch_size
            padded_spatial_shape = next(iter(padded_volumes.values())).shape[1:]
            if torch.any(start_coord < 0) or torch.any(end_coord > torch.tensor(padded_spatial_shape, device=center_coord.device)):
                continue
            patches = {}
            for name, padded_volume in padded_volumes.items():
                if spatial_dims == 2:
                    patch = padded_volume[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
                else:
                    patch = padded_volume[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]]
                patches[name] = patch
            first_patch = next(iter(patches.values()))
            if first_patch.abs().max() > 0:
                return patches
        return self._sample_uniform_patch(padded_volumes, spatial_dims)

    def _sample_uniform_patch(self, padded_volumes: Dict[str, torch.Tensor], spatial_dims: int) -> Dict[str, torch.Tensor]:
        """Sample one patch uniformly within the spatial bounds."""
        spatial_shape = next(iter(padded_volumes.values())).shape[1:]
        max_start = torch.tensor(spatial_shape, device=next(iter(padded_volumes.values())).device) - self.patch_size
        max_start = torch.clamp(max_start, min=1)
        if spatial_dims == 2:
            start_h = torch.randint(0, max_start[0].item(), (1,), device=next(iter(padded_volumes.values())).device)
            start_w = torch.randint(0, max_start[1].item(), (1,), device=next(iter(padded_volumes.values())).device)
        else:
            start_d = torch.randint(0, max_start[0].item(), (1,), device=next(iter(padded_volumes.values())).device)
            start_h = torch.randint(0, max_start[1].item(), (1,), device=next(iter(padded_volumes.values())).device)
            start_w = torch.randint(0, max_start[2].item(), (1,), device=next(iter(padded_volumes.values())).device)
        patches = {}
        for name, padded_volume in padded_volumes.items():
            if spatial_dims == 2:
                patch = padded_volume[:, start_h:start_h + self.patch_size[0], start_w:start_w + self.patch_size[1]]
            else:
                patch = padded_volume[:, start_d:start_d + self.patch_size[0], start_h:start_h + self.patch_size[1], start_w:start_w + self.patch_size[2]]
            patches[name] = patch
        return patches
