from skimage import filters, morphology
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import pickle
import SimpleITK as sitk
import numpy as np

def do_preprocess(image_fd, fd_lst, dst_fd_lst, mask_0_threshold=20):
    assert len(fd_lst) == len(dst_fd_lst), 'The length of source and destination folder lists must match.'
    for fd, dst_fd in zip(fd_lst, dst_fd_lst):
        image_path = osp.join(image_fd, fd)
        dst_path = osp.join(image_fd, dst_fd)
        if not osp.exists(dst_path):
            os.makedirs(dst_path)
        for name in sorted(os.listdir(image_path)):
            if not osp.isdir(osp.join(image_path, name)):
                continue
            image = cv2.imread(osp.join(image_path, name, 'image.png'))
            if image is None:
                raise FileNotFoundError(osp.join(image_path, name, 'image.png'))
            mask_0 = image.sum(axis=2) > mask_0_threshold
            mask_0 = morphology.binary_closing(mask_0, morphology.disk(25))
            mask_0 = morphology.binary_erosion(mask_0, morphology.disk(20))
            image = image / 255.0
            image = image[:, :, 1]
            image_shape_ori = image.shape
            image_small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            frangi = filters.frangi(image_small, sigmas=[1, 2, 4, 6, 8, 12], alpha=0.1, beta=0.1, black_ridges=True)
            frangi = cv2.resize(frangi, image_shape_ori[::-1], interpolation=cv2.INTER_LINEAR)
            frangi = frangi * mask_0
            normalizer = float(np.mean(frangi[mask_0])) if mask_0.any() else 0.0
            if normalizer <= 0 or not np.isfinite(normalizer):
                raise ValueError(f'Empty field of view or zero Frangi response: {name}')
            frangi = frangi / normalizer
            pickle.dump({'arr': frangi}, open(osp.join(dst_path, f'{name}.pt'), 'wb'))
            plt.imsave(osp.join(dst_path, f'{name}.png'), frangi, cmap='gray', vmin=0, vmax=20)
            plt.imsave(osp.join(dst_path, f'{name}_mask_0.png'), mask_0, cmap='gray')
            sitk.WriteImage(sitk.GetImageFromArray(frangi[np.newaxis, :, :]), osp.join(dst_path, f'{name}.nii.gz'))
            print('Finish processing {}'.format(name))


if __name__ == '__main__':
    import argparse
    from utils import load_yaml
    from submission_config import dataset_root
    parser = argparse.ArgumentParser(description='Extract normalized Frangi vesselness priors.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    args = parser.parse_args()
    config = load_yaml(args.config)
    do_preprocess(str(dataset_root(config)), args.splits, ['frangi_' + s for s in args.splits],
                  mask_0_threshold=config.get('preprocessing', {}).get('fov_threshold', 20))
