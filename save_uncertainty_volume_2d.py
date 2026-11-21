import torch
import torch.nn as nn
import os
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from dataset import Dataset2d
from trainer import WrappedModelBase, Trainer2d
from utils import define_fd
from baseline_plpb import get_mask_from_uncertainty_2d
from baseline_tt_sfuda import uncert_voting_2d, uncert_voting_2d_use_pseudo, build_pseduo_augmentation_2d, build_pseduo_augmentation_plpb_2d, build_pseudo_augmentation_flip_2d, flip_back_2d_flip, flip_back_plpb_2d, build_pseudo_augmentation_flip_with_gray_2d, build_pseudo_augmentation_flip_with_gray_2d_v2, flip_back_2d_flip_with_gray_v2
from exp_gnn_pseudo_learn_2d import choose_dataset, make_net_2d

class WrappedModel(WrappedModelBase):

    def __init__(self, model, device, aug_type, pseudo_mode, mode='mean'):
        super().__init__(model)
        self.device = device
        self.mode = mode
        self.aug_type = aug_type
        self.pseudo_mode = pseudo_mode

    def forward(self, x):
        self.model.train()
        pred_lst = []
        features_lst = []
        x = x.to(device=self.device, dtype=torch.float32)
        if self.aug_type == 'orig':
            augs_x = build_pseduo_augmentation_2d(x)
        elif self.aug_type == 'flip':
            augs_x = build_pseduo_augmentation_plpb_2d(x.cpu().numpy())
        elif self.aug_type == 'flip_only':
            augs_x = build_pseudo_augmentation_flip_2d(x)
        elif self.aug_type == 'flip_with_gray':
            augs_x = build_pseudo_augmentation_flip_with_gray_2d(x)
        elif self.aug_type == 'flip_with_gray_v2':
            augs_x = build_pseudo_augmentation_flip_with_gray_2d_v2(x)
        else:
            raise ValueError('Invalid aug_type')
        assert augs_x.shape[0] == 1
        with torch.no_grad():
            x_aug = augs_x[0].to(device=self.device, dtype=torch.float32)
            pred, leak_dic = self.model(x_aug)
            if self.pseudo_mode == 'plpb':
                pred = torch.sigmoid(pred)
        feature = leak_dic['last_feature_map']
        features_lst.append(feature.cpu())
        pred_lst = torch.clone(pred)
        del pred
        if self.mode == 'mean':
            pred = pred_lst.mean(dim=0, keepdim=True)
            if self.pseudo_mode != 'plpb':
                pred = torch.sigmoid(pred)
        elif self.mode == 'max':
            pred = pred_lst.max(dim=0, keepdim=True).values
            if self.pseudo_mode != 'plpb':
                pred = torch.sigmoid(pred)
        std = pred_lst.std(dim=0, keepdim=True)
        feature = torch.stack(features_lst).mean(dim=0)
        if self.aug_type == 'flip':
            pred_lst = flip_back_plpb_2d(pred_lst)
            if self.pseudo_mode != 'plpb':
                feature = torch.zeros_like(feature)
            else:
                feature = flip_back_plpb_2d(feature)
        elif self.aug_type in ['flip_only', 'flip_with_gray']:
            pred_lst = flip_back_2d_flip(pred_lst)
            feature = flip_back_2d_flip(feature)
        elif self.aug_type == 'flip_with_gray_v2':
            pred_lst = flip_back_2d_flip_with_gray_v2(pred_lst)
            feature = flip_back_2d_flip_with_gray_v2(feature)
        return {'pred': pred, 'std': std, 'feature': feature, 'preds': pred_lst}

class TrainerSave(Trainer2d):

    def __init__(self, model, optimizer, criterion, device, train_loader, test_loader, pred_mode, **kwargs):
        super().__init__(model, optimizer, criterion, device, train_loader, test_loader, **kwargs)
        self.pred_mode = pred_mode
        self.pseudo_mode = kwargs['pseudo_mode']
        self.pseudo_pseudo_thres = kwargs.get('pseudo_pseudo_thres', 0.15)
        self.aug_type = kwargs['aug_type']

    def test(self, loader, save_path, loader_name='Test', model_path=None, thres=0.01, std_thres=0.05):
        self.pred_path = save_path
        self.model.train()
        if model_path is not None:
            self.model.load_state_dict(torch.load(osp.join(self.base_folder, model_path)), strict=True)
        idx = 0
        if not osp.exists(osp.join(self.pred_path, loader_name)):
            os.makedirs(osp.join(self.pred_path, loader_name))
        with tqdm(loader) as t:
            for batch in t:
                idx += 1
                image = batch['image']
                image = self._normalize(image)
                wrapped_model = self._wrap_model_forward_eval()
                with torch.no_grad():
                    dic = wrapped_model(image)
                pred = dic['pred']
                std = dic['std']
                feature = dic['feature']
                preds = dic['preds'].detach().cpu()
                pred = torch.cat([1 - pred, pred], dim=1)
                std = torch.cat([std, std], dim=1)
                if self.pseudo_mode == 'plpb':
                    mask, pseudo_label = get_mask_from_uncertainty_2d(pred, std, feature, pred_thres=thres, std_thres=std_thres)
                elif self.pseudo_mode == 'tt_sfuda':
                    pseudo_label, mask, _ = uncert_voting_2d(preds)
                elif self.pseudo_mode == 'pseudo':
                    pseudo_label, mask, _ = uncert_voting_2d_use_pseudo(preds, thres=self.pseudo_pseudo_thres)
                else:
                    raise ValueError('Invalid pseudo_mode')
                pickle.dump(mask, open(osp.join(self.pred_path, loader_name, batch['name'][0] + '_mask.pkl'), 'wb'))
                plt.imsave(osp.join(self.pred_path, loader_name, batch['name'][0] + '_pseudo.png'), pseudo_label[0, 0].numpy(), cmap='gray')
                plt.imsave(osp.join(self.pred_path, loader_name, batch['name'][0] + '_mask.png'), mask[0, 0].numpy(), cmap='gray')
                pickle.dump(pseudo_label, open(osp.join(self.pred_path, loader_name, batch['name'][0] + '_pseudo.pkl'), 'wb'))
                pickle.dump(pred.detach().cpu(), open(osp.join(self.pred_path, loader_name, batch['name'][0] + '_pred.pkl'), 'wb'))
                print('Save mask to {}'.format(osp.join(self.pred_path, loader_name, batch['name'][0] + '.pkl')))

    def _wrap_model_forward_eval(self):
        return WrappedModel(self.model, self.device, self.aug_type, self.pseudo_mode, mode=self.pred_mode)

def main():
    import argparse
    from pathlib import Path
    from utils import load_yaml, write_yaml, set_all_seeds
    from submission_config import configure_runtime, dataset_root, cases
    parser = argparse.ArgumentParser(description='Generate source-model pseudo labels and uncertainty maps.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    args = parser.parse_args()
    config = load_yaml(args.config)
    configure_runtime(config)
    set_all_seeds(config.get('seed', 42))
    pp = config.get('preprocessing', {})
    size = tuple(config.get('image_size', [1024, 1024]))
    device = torch.device(config.get('device', 'cuda:0'))
    model = make_net_2d(norm_layer=config.get('net_norm_method', 'batch'), image_size=size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    root = dataset_root(config)
    log_dir = root / 'preprocessing_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(str(log_dir / 'pseudo_config.yaml'), config)
    # Inherited source preprocessing uses train-mode BN with frozen running statistics.
    # Each split contains only images; no target label is used by TrainerSave.test.
    trainer = TrainerSave(model, optimizer, None, device, None, None, pred_mode='mean',
                          base_folder=str(log_dir), patch_size=size, max_epoch=1,
                          pseudo_mode=pp.get('pseudo_mode', 'tt_sfuda'),
                          aug_type=pp.get('aug_type', 'flip_with_gray_v2'),
                          pseudo_pseudo_thres=pp.get('pseudo_threshold', 0.15))
    for split in args.splits:
        cases(config, split)
        data = Dataset2d(str(root / split), mode='test', norm_method={'name': 'png_norm'},
                         wanted_size=size, allow_no_label=True, only_no_label=True, augs=None)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=config.get('num_workers', 0))
        suffix = 'train' if split == 'train' else 'test'
        trainer.test(loader, str(root), loader_name=f'PLPB_mask_{suffix}', thres=0.5, std_thres=0.05)


if __name__ == '__main__':
    main()
