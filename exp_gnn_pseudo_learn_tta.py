from submission_config import configure_runtime
import argparse
import copy
import csv
import os
import os.path as osp
import numpy as np
import torch
import torchio as tio
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CustomDataset
from exp_gnn_pseudo_learn import TrainerGNN, choose_dataset, get_points_and_edges_within_patch, make_net
from metrics import dice_torch
from trainer import calc_metrics_torch
from utils import TxtLogger, dict_element_add, dict_element_divide, dict_element_lst_append, load_yaml, write_yaml
from utils_preprocess import threshold_torch

class TrainerGNNTTA3D(TrainerGNN):

    @staticmethod
    def _unwrap_model_state_dict(state_dict):
        if isinstance(state_dict, dict) and 'network_weights' in state_dict:
            return state_dict['network_weights']
        return state_dict

    def cache_tta_source_state(self, use_model_as_teacher=True):
        if use_model_as_teacher:
            self.orig_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.teacher_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self._tta_source_state = {'model': copy.deepcopy(self.model.state_dict()), 'orig_model': copy.deepcopy(self.orig_model.state_dict()), 'teacher_model': copy.deepcopy(self.teacher_model.state_dict()), 'gnn': copy.deepcopy(self.gnn.state_dict()), 'optimizer': copy.deepcopy(self.optimizer.state_dict()), 'optimizer_gnn': copy.deepcopy(self.optimizer_gnn.state_dict())}
        if getattr(self, 'refiner', None) is not None:
            self._tta_source_state['refiner'] = copy.deepcopy(self.refiner.state_dict())
            self._tta_source_state['optimizer_refiner'] = copy.deepcopy(self.optimizer_refiner.state_dict())

    def configure_tta_reinitialization(self, model_path=None, gnn_path=None, refiner_path=None, use_model_as_teacher=True):
        self._tta_model_init_path = model_path
        self._tta_gnn_init_path = gnn_path
        self._tta_refiner_init_path = refiner_path
        self._tta_use_model_as_teacher = use_model_as_teacher
        self.cache_tta_source_state(use_model_as_teacher=use_model_as_teacher)

    def _reload_initial_model_for_tta(self):
        if getattr(self, '_tta_model_init_path', None):
            state_dict = self._unwrap_model_state_dict(torch.load(self._tta_model_init_path, map_location=torch.device('cpu')))
        else:
            fresh_model = make_net(net_conf=self.kwargs.get('tta_net_conf', 'large'), track=self.kwargs.get('tta_track', False))
            state_dict = fresh_model.state_dict()
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device=self.device)
        if getattr(self, '_tta_use_model_as_teacher', True):
            self.orig_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.teacher_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def reset_to_initial_weights_for_tta(self):
        if not hasattr(self, '_tta_source_state'):
            self.configure_tta_reinitialization(use_model_as_teacher=self.kwargs.get('tta_use_model_as_teacher', True))
        self._reload_initial_model_for_tta()
        if not getattr(self, '_tta_use_model_as_teacher', True):
            self.orig_model.load_state_dict(copy.deepcopy(self._tta_source_state['orig_model']))
            self.teacher_model.load_state_dict(copy.deepcopy(self._tta_source_state['teacher_model']))
        if getattr(self, '_tta_gnn_init_path', None):
            self.gnn.load_state_dict(torch.load(self._tta_gnn_init_path, map_location=torch.device('cpu')), strict=False)
        else:
            self.gnn.load_state_dict(copy.deepcopy(self._tta_source_state['gnn']))
        if getattr(self, 'refiner', None) is not None and 'refiner' in self._tta_source_state:
            if getattr(self, '_tta_refiner_init_path', None) and osp.exists(self._tta_refiner_init_path):
                self.refiner.load_state_dict(torch.load(self._tta_refiner_init_path, map_location=torch.device('cpu')), strict=False)
            else:
                self.refiner.load_state_dict(copy.deepcopy(self._tta_source_state['refiner']))
        self.optimizer.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer']))
        self.optimizer_gnn.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer_gnn']))
        if getattr(self, 'refiner', None) is not None and 'optimizer_refiner' in self._tta_source_state:
            self.optimizer_refiner.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer_refiner']))
        self.optimizer.zero_grad()
        self.optimizer_gnn.zero_grad()
        if getattr(self, 'refiner', None) is not None:
            self.optimizer_refiner.zero_grad()

    def _batch_to_volume_inputs(self, batch):
        image = batch['image'].to(device=self.device, dtype=torch.float32)
        image = self._normalize(image)
        frangi = batch['frangi'].to(device=self.device, dtype=torch.float32)
        gnn_dic = batch['gnn']
        gnn2_dic = batch.get('gnn2', None)
        gnn_dic_list = [gnn_dic, gnn2_dic] if gnn2_dic is not None else [gnn_dic]
        return (image, frangi, gnn_dic_list)

    def _sample_tta_patches(self, batch):
        image = batch['image']
        distmap = batch['frangi']
        uncertainty_mask = batch['uncertainty_mask'][0][0]
        pseudo_label = batch['pseudo_label'][0][0]
        if 'area_interest' in batch:
            label_for_sampling = batch['area_interest']
        else:
            label_for_sampling = ((pseudo_label > 0) | (1 - uncertainty_mask > 0)).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gnn_dic = batch['gnn']
        gnn2_dic = batch.get('gnn2', None)
        distmap_shifted = torch.zeros_like(distmap)
        distmap_shifted[:, :, 1:] = distmap[:, :, :-1]
        image = self._normalize(image)
        subject = tio.Subject(image=tio.ScalarImage(tensor=image.squeeze(0)), label=tio.LabelMap(tensor=label_for_sampling.squeeze(0)), distmap=tio.ScalarImage(tensor=distmap_shifted.squeeze(0)), uncertainty_mask=tio.LabelMap(tensor=uncertainty_mask.squeeze(0)), pseudo_label=tio.LabelMap(tensor=pseudo_label.squeeze(0)))
        samples = self.sampler(subject, self.patches_in_batch)
        patch_inputs = []
        for sample in samples:
            image_patch = sample['image']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32)
            frangi_patch = sample['distmap']['data'].unsqueeze(0).to(device=self.device, dtype=torch.float32)
            gnn_dic_patch = get_points_and_edges_within_patch(gnn_dic, sample['location'], self.device, self.logger)
            gnn_dic_list = [gnn_dic_patch]
            if gnn2_dic is not None:
                gnn2_dic_patch = get_points_and_edges_within_patch(gnn2_dic, sample['location'], self.device, self.logger)
                gnn_dic_list.append(gnn2_dic_patch)
            patch_inputs.append((image_patch, frangi_patch, gnn_dic_list))
        return patch_inputs

    def _do_tta_patch_train_step(self, batch, epoch):
        total_losses = {}
        patch_inputs = self._sample_tta_patches(batch)
        for image_patch, frangi_patch, gnn_dic_list in patch_inputs:
            losses_dic = self._do_train_tt_sfuda(self.model, self.orig_model, self.gnn, image_patch, gnn_dic_list, epoch, frangi=frangi_patch)
            total_losses = dict_element_add(total_losses, losses_dic)
        return dict_element_divide(total_losses, len(patch_inputs))

    def _predict_adapted_batch(self, image, thres):
        self._set_eval()
        wrapped_model = self._wrap_model_forward_eval(model_to_test='model')
        with torch.no_grad():
            pred = sliding_window_inference(image, self.patch_size, self.test_batch_size, wrapped_model)
        pred = pred.cpu().to(dtype=torch.float32)
        pred = threshold_torch(pred, thres)
        return self._postprocess(pred)


    def _calc_batch_metrics(self, pred, label, batch):
        rows = []
        for i in range(pred.shape[0]):
            pred_i = pred[i:i + 1]
            label_i = label[i:i + 1]
            if pred_i.shape != label_i.shape:
                raise ValueError(f'Prediction shape {pred_i.shape} does not match label shape {label_i.shape}')
            dice = dice_torch(pred_i, label_i)
            dice = float(dice.item() if hasattr(dice, 'item') else dice)
            metrics = calc_metrics_torch(pred_i, label_i)
            rows.append((batch['name'][i], dice, metrics))
        return rows

    def _save_step_prediction(self, pred, batch, loader_name, step):
        step_loader_name = osp.join(loader_name, 'step_preds')
        save_dir = osp.join(self.pred_path, step_loader_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        pred_np = pred.squeeze(0).squeeze(0).numpy()
        self._do_save_image(batch, pred_np, step_loader_name, postfix=f'_step_{step:04d}_pred')

    def tta_test(self, loader, loader_name='TTA', tta_steps=10, thres=0.5, max_cases=None, save_pred_interval=10, gnn_distillation_iters=0):
        if not hasattr(self, '_tta_source_state'):
            self.configure_tta_reinitialization(use_model_as_teacher=self.kwargs.get('tta_use_model_as_teacher', True))
        save_dir = osp.join(self.pred_path, loader_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        logger_new = TxtLogger(osp.join(save_dir, 'results.txt'), verbose=True)
        original_consistency_start_epoch = self.kwargs['gnn'].get('consistency_start_epoch', None)
        had_consistency_start_epoch = 'consistency_start_epoch' in self.kwargs['gnn']
        self.kwargs['gnn']['consistency_start_epoch'] = 10 ** 12
        logger_new.write(f'GNN distillation/consistency ignores config consistency_start_epoch. Extra GNN distillation iterations: {gnn_distillation_iters}.')
        total_metrics = {}
        total_adapt_steps = tta_steps + max(gnn_distillation_iters, 0)
        try:
            with tqdm(loader) as t:
                for case_idx, batch in enumerate(t):
                    if max_cases is not None and case_idx >= max_cases:
                        break
                    self.reset_to_initial_weights_for_tta()
                    image, _, _ = self._batch_to_volume_inputs(batch)
                    name = batch['name']
                    label = batch['label'].cpu().to(dtype=torch.float32)
                    self.model.train()
                    self.gnn.train()
                    for step in range(tta_steps):
                        loss_dic = self._do_tta_patch_train_step(batch, step)
                        step_num = step + 1
                        should_save = save_pred_interval is not None and save_pred_interval > 0 and (step_num % save_pred_interval == 0)
                        if should_save:
                            pred_step = self._predict_adapted_batch(image, thres)
                            if should_save:
                                self._save_step_prediction(pred_step, batch, loader_name, step_num)
                        else:
                            t.set_postfix({'case': name[0], 'step': step_num})
                        self.model.train()
                        self.gnn.train()
                    if gnn_distillation_iters > 0:
                        self.kwargs['gnn']['consistency_start_epoch'] = 0
                        logger_new.write(f'Start {gnn_distillation_iters} extra GNN distillation steps for {name[0]}.')
                    for distill_step in range(max(gnn_distillation_iters, 0)):
                        epoch_for_loss = tta_steps + distill_step
                        loss_dic = self._do_tta_patch_train_step(batch, epoch_for_loss)
                        step_num = tta_steps + distill_step + 1
                        should_save = save_pred_interval is not None and save_pred_interval > 0 and (step_num % save_pred_interval == 0)
                        if should_save:
                            pred_step = self._predict_adapted_batch(image, thres)
                            if should_save:
                                self._save_step_prediction(pred_step, batch, loader_name, step_num)
                        else:
                            t.set_postfix({'case': name[0], 'step': step_num, 'phase': 'gnn_distill'})
                        self.model.train()
                        self.gnn.train()
                    self.kwargs['gnn']['consistency_start_epoch'] = 10 ** 12
                    pred = self._predict_adapted_batch(image, thres)
                    metric_rows = self._calc_batch_metrics(pred, label, batch)
                    for sample_name, dice, metrics in metric_rows:
                        logger_new.write(f"TTA final Dice for {sample_name}: {dice:.4f}. Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                        total_metrics = dict_element_lst_append(total_metrics, metrics)
                    pred_np = pred.squeeze(0).squeeze(0).numpy()
                    self._do_save_image(batch, pred_np, loader_name, postfix='_pred')
        finally:
            if had_consistency_start_epoch:
                self.kwargs['gnn']['consistency_start_epoch'] = original_consistency_start_epoch
            else:
                self.kwargs['gnn'].pop('consistency_start_epoch', None)
        total_metrics_mean = {}
        total_metrics_std = {}
        for key, value in total_metrics.items():
            if isinstance(value, list):
                total_metrics_mean[key] = np.mean(value)
                total_metrics_std[key] = np.std(value)
            else:
                total_metrics_mean[key] = value
                total_metrics_std[key] = 0.0
        test_str = 'TTA test result: '
        for key, value in total_metrics_mean.items():
            test_str += f'{key}: {value:.4f} +- {total_metrics_std[key]:.4f}, '
        logger_new.write(test_str)
        self.logger.write(test_str)
        return total_metrics_mean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp_gnn_pseudo_learn/tof_to_snap.yaml')
    parser.add_argument('--tta_steps', type=int, default=1000)
    parser.add_argument('--gnn_distillation_iters', type=int, default=500)
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'unlabeled', 'all_trmod'])
    parser.add_argument('--base_folder', type=str, default=None)
    parser.add_argument('--loader_name', type=str, default='TTA')
    parser.add_argument('--thres', type=float, default=0.5)
    parser.add_argument('--save_pred_interval', type=int, default=10)
    parser.add_argument('--trained_path', type=str, default=None)
    parser.add_argument('--pretrain_gnn_path', type=str, default=None)
    parser.add_argument('--keep_config_teacher', action='store_true')
    return parser.parse_args()

def apply_patch_size_description(full_config):
    patch_size_description = full_config['trainer_configs'].get('patch_size_description')
    if patch_size_description == 'large':
        full_config['trainer_configs']['patch_size'] = [64, 320, 256]
    elif patch_size_description == 'small':
        full_config['trainer_configs']['patch_size'] = [40, 224, 192]
    return patch_size_description

def build_tta_loader(full_config, split='test'):
    snap_data_path, invert_grayscale, norm_method, frangi_path, default_graph_side_len, other_dic = choose_dataset(full_config)
    graph_version = full_config.get('graph_version', 2)
    highres_graph_sidelen = full_config.get('highres_graph_sidelen', default_graph_side_len[0])
    lowres_graph_sidelen = full_config.get('lowres_graph_sidelen', default_graph_side_len[1])
    trainer_configs = full_config['trainer_configs']
    other_aux = [{'name': 'gnn', 'aux_path': snap_data_path + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}', 'type': 'pkl'}]
    if trainer_configs['gnn'].get('scales', 1) >= 2:
        other_aux.append({'name': 'gnn2', 'aux_path': snap_data_path + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}', 'type': 'pkl'})
    last_stage_mask_path = {'frangi': frangi_path}
    if other_dic.get('pseudo_path', None) is not None and other_dic.get('mask_path', None) is not None:
        last_stage_mask_path['pseudo'] = other_dic['pseudo_path']
        last_stage_mask_path['mask'] = other_dic['mask_path']
    if split == 'test':
        allow_no_label = False
        only_no_label = False
    elif split == 'unlabeled':
        allow_no_label = True
        only_no_label = True
    elif split == 'all_trmod':
        allow_no_label = True
        only_no_label = False
    else:
        raise ValueError(f'Invalid split {split}')
    dataset = CustomDataset(snap_data_path, mode='test', test_fd_str='TrMod', invert_grayscale=invert_grayscale, allow_no_label=allow_no_label, only_no_label=only_no_label, last_stage_mask_path=last_stage_mask_path, other_aux=other_aux, load_mask_pseudo=snap_data_path + '/PLPB_mask', norm_method=norm_method, wanted_names=(osp.join(snap_data_path, other_dic['wanted_names'][0]), other_dic['wanted_names'][1]) if 'wanted_names' in other_dic else None)
    return DataLoader(dataset, batch_size=1, shuffle=False)

def build_optimizer(model, training_configs):
    if training_configs['optimizer_type'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    if training_configs['optimizer_type'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    if training_configs['optimizer_type'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], momentum=0.9)
    if training_configs['optimizer_type'] == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    raise ValueError(f"Invalid optimizer type {training_configs['optimizer_type']}")

def main():
    args = parse_args()
    full_config = load_yaml(args.config)
    configure_runtime(full_config)
    patch_size_description = apply_patch_size_description(full_config)
    trainer_configs = full_config['trainer_configs']
    training_configs = full_config['training_configs']
    trainer_configs.update(training_configs)
    trainer_configs['training_base'] = 'tt_sfuda'
    trainer_configs['tta_use_model_as_teacher'] = not args.keep_config_teacher
    trainer_configs['tta_net_conf'] = patch_size_description if patch_size_description is not None else 'large'
    trainer_configs['tta_track'] = full_config.get('track', False)
    loader = build_tta_loader(full_config, split=args.split)
    run_name = full_config.get('run_name', 'vesselgda')
    base_folder = args.base_folder or osp.join('results', run_name + '_tta')
    if not osp.exists(base_folder):
        os.makedirs(base_folder)
    write_yaml(osp.join(base_folder, 'config.yaml'), full_config)
    device = torch.device(full_config.get('device', 'cuda:0'))
    model = make_net(net_conf=trainer_configs['tta_net_conf'], track=trainer_configs['tta_track'])
    if args.trained_path or trainer_configs.get('trained_path'):
        state_dict = TrainerGNNTTA3D._unwrap_model_state_dict(torch.load(args.trained_path or trainer_configs['trained_path'], map_location=torch.device('cpu')))
        model.load_state_dict(state_dict, strict=False)
    optimizer = build_optimizer(model, training_configs)
    model = model.to(device)
    trainer = TrainerGNNTTA3D(model, optimizer, None, device, loader, loader, base_folder=base_folder, set_to_eval_when_testing=True, **trainer_configs)
    tta_gnn_path = args.pretrain_gnn_path or trainer_configs.get('pretrain_gnn_path')
    tta_refiner_path = None
    if tta_gnn_path:
        trainer.gnn.load_state_dict(torch.load(tta_gnn_path, map_location=torch.device('cpu')), strict=False)
        tta_refiner_path = tta_gnn_path.replace('gnn', 'refiner')
        if getattr(trainer, 'refiner', None) is not None and osp.exists(tta_refiner_path):
            trainer.refiner.load_state_dict(torch.load(tta_refiner_path, map_location=torch.device('cpu')), strict=False)
    trainer.configure_tta_reinitialization(model_path=args.trained_path or trainer_configs.get('trained_path'), gnn_path=tta_gnn_path, refiner_path=tta_refiner_path, use_model_as_teacher=trainer_configs['tta_use_model_as_teacher'])
    tta_steps = args.tta_steps if args.tta_steps is not None else trainer_configs.get('tta_steps', trainer_configs['max_epoch'])
    loader_name = args.loader_name
    if args.gnn_distillation_iters > 0:
        loader_name = f'{loader_name}_gnn_distill_{args.gnn_distillation_iters}'
    trainer.tta_test(loader, loader_name=loader_name, tta_steps=tta_steps, thres=args.thres, max_cases=args.max_cases, save_pred_interval=args.save_pred_interval, gnn_distillation_iters=args.gnn_distillation_iters)
if __name__ == '__main__':
    main()
