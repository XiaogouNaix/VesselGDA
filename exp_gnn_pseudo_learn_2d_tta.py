from submission_config import configure_runtime
import argparse
import copy
import csv
import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Dataset2d
from exp_gnn_pseudo_learn_2d import TrainerGNN, choose_dataset, custom_collate_fn, make_net_2d
from metrics import dice_torch
from trainer import calc_metrics_torch
from utils import TxtLogger, dict_element_lst_append, load_yaml, set_all_seeds, write_yaml
from utils_preprocess import threshold_torch

class TrainerGNNTTA(TrainerGNN):
    """
    Test-time adaptation wrapper for the 2D GNN pseudo-label learner.

    The inherited _do_train_tt_sfuda method already performs one adaptation
    update for one input batch. This class changes only the outer loop:
    reset to the source model for each test case, repeat adaptation on that
    same case, then evaluate that case.
    """

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
            state_dict = torch.load(self._tta_model_init_path, map_location=torch.device('cpu'))
        else:
            fresh_model = make_net_2d(net_conf=self.kwargs.get('tta_net_conf', 'FIVES'), norm_layer=self.kwargs.get('tta_norm_layer', 'batch'), image_size=self.kwargs.get('tta_image_size', (1024, 1024)))
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

    def reset_to_tta_source_state(self):
        if not hasattr(self, '_tta_source_state'):
            self.cache_tta_source_state()
        self.model.load_state_dict(copy.deepcopy(self._tta_source_state['model']))
        self.orig_model.load_state_dict(copy.deepcopy(self._tta_source_state['orig_model']))
        self.teacher_model.load_state_dict(copy.deepcopy(self._tta_source_state['teacher_model']))
        self.gnn.load_state_dict(copy.deepcopy(self._tta_source_state['gnn']))
        self.optimizer.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer']))
        self.optimizer_gnn.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer_gnn']))
        if 'refiner' in self._tta_source_state:
            self.refiner.load_state_dict(copy.deepcopy(self._tta_source_state['refiner']))
            self.optimizer_refiner.load_state_dict(copy.deepcopy(self._tta_source_state['optimizer_refiner']))

    def _batch_to_tta_inputs(self, batch):
        image = batch['image'].to(device=self.device, dtype=torch.float32)
        image = self._normalize(image)
        frangi = batch['frangi'].to(device=self.device, dtype=torch.float32)
        gnn_dic = batch['gnn']
        gnn2_dic = batch.get('gnn2', None)
        gnn_dic_list = [gnn_dic, gnn2_dic] if gnn2_dic is not None else [gnn_dic]
        return (image, frangi, gnn_dic_list)

    def _predict_adapted_batch(self, image, thres):
        self._set_eval()
        wrapped_model = self._wrap_model_forward_eval(model_to_test='model')
        with torch.no_grad():
            if image.shape[-2] <= self.patch_size[-2] and image.shape[-1] <= self.patch_size[-1]:
                pred = wrapped_model(image)
            else:
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
        save_dir = osp.join(self.pred_path, loader_name, 'step_preds')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(pred.shape[0]):
            pred_np = pred[i].squeeze().numpy()
            save_path = osp.join(save_dir, f"{batch['name'][i]}_step_{step:04d}_pred.png")
            plt.imsave(save_path, pred_np, cmap='gray')

    def _final_prediction_path(self, loader_name, sample_name):
        return osp.join(self.pred_path, loader_name, f'{sample_name}_pred.png')

    def _completion_marker_path(self, loader_name, sample_name):
        return osp.join(self.pred_path, loader_name, '.tta_complete', f'{sample_name}.json')

    @staticmethod
    def _is_complete_png(path):
        """Reject an output file that was left truncated while being written."""
        if not osp.isfile(path) or osp.getsize(path) < 20:
            return False
        with open(path, 'rb') as f:
            if f.read(8) != b'\x89PNG\r\n\x1a\n':
                return False
            f.seek(-12, os.SEEK_END)
            return f.read(12) == b'\x00\x00\x00\x00IEND\xaeB`\x82'

    def _case_is_complete(self, loader_name, sample_names):
        """Final PNGs also make this compatible with runs made before markers existed."""
        for sample_name in sample_names:
            pred_path = self._final_prediction_path(loader_name, sample_name)
            if not self._is_complete_png(pred_path):
                return False
        return True

    def _mark_case_complete(self, loader_name, sample_names, tta_steps, gnn_distillation_iters):
        marker_dir = osp.join(self.pred_path, loader_name, '.tta_complete')
        os.makedirs(marker_dir, exist_ok=True)
        for sample_name in sample_names:
            marker_path = self._completion_marker_path(loader_name, sample_name)
            tmp_path = marker_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump({'name': sample_name, 'tta_steps': tta_steps, 'gnn_distillation_iters': gnn_distillation_iters, 'prediction': self._final_prediction_path(loader_name, sample_name)}, f, indent=2)
            os.replace(tmp_path, marker_path)

    def tta_test(self, loader, loader_name='TTA', tta_steps=10, thres=0.5, max_cases=None, save_pred_interval=10, gnn_distillation_iters=0, resume=True):
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
        skipped_cases = 0
        total_adapt_steps = tta_steps + max(gnn_distillation_iters, 0)
        try:
            with tqdm(loader) as t:
                for case_idx, batch in enumerate(t):
                    if max_cases is not None and case_idx >= max_cases:
                        break
                    sample_names = list(batch['name'])
                    if resume and self._case_is_complete(loader_name, sample_names):
                        skipped_cases += 1
                        logger_new.write(f"Skip completed TTA case(s): {', '.join(sample_names)}")
                        t.set_postfix({'case': sample_names[0], 'status': 'skipped'})
                        continue
                    self.reset_to_initial_weights_for_tta()
                    image, frangi, gnn_dic_list = self._batch_to_tta_inputs(batch)
                    name = batch['name']
                    label = batch['label'].cpu().to(dtype=torch.float32)
                    self.model.train()
                    self.gnn.train()
                    for step in range(tta_steps):
                        loss_dic = self._do_train_tt_sfuda(self.model, self.orig_model, self.gnn, image, gnn_dic_list, step, name, frangi=frangi)
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
                        loss_dic = self._do_train_tt_sfuda(self.model, self.orig_model, self.gnn, image, gnn_dic_list, epoch_for_loss, name, frangi=frangi)
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
                    self._mark_case_complete(loader_name, sample_names, tta_steps, gnn_distillation_iters)
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
        if resume:
            logger_new.write(f'TTA resume skipped {skipped_cases} completed case(s).')
        self.logger.write(test_str)
        return total_metrics_mean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml')
    parser.add_argument('--tta_steps', type=int, default=1000)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train'])
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--base_folder', type=str, default=None)
    parser.add_argument('--loader_name', type=str, default='TTA')
    parser.add_argument('--thres', type=float, default=0.5)
    parser.add_argument('--save_pred_interval', type=int, default=10)
    parser.add_argument('--gnn_distillation_iters', type=int, default=500)
    parser.add_argument('--keep_config_teacher', action='store_true')
    parser.add_argument('--no_resume', action='store_true', help='Rerun all selected cases even when a final TTA prediction already exists.')
    return parser.parse_args()

def build_loaders(full_config):
    data_path_base, data_path_train, data_path_test, norm_method, frangi_path = choose_dataset(full_config)
    frangi_path_test = osp.join(data_path_base, 'frangi_val')
    graph_version = full_config.get('graph_version', 2)
    highres_graph_sidelen = full_config.get('highres_graph_sidelen', 16)
    lowres_graph_sidelen = full_config.get('lowres_graph_sidelen', 32)
    graph_masked_str = '_masked' if full_config.get('graph_masked', False) else ''
    graph_psuncert_str = '_psuncert' if full_config.get('graph_psuncert', False) else ''
    graph_trained_num_str = f"_{full_config['graph_trained_num']}" if full_config.get('graph_trained_num') is not None else ''
    trainer_configs = full_config['trainer_configs']
    if graph_version >= 10000:
        trainer_configs['gnn']['use_slic'] = True
    other_aux = [{'name': 'gnn', 'aux_path': data_path_base + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_train', 'type': 'pkl'}]
    other_aux_test = [{'name': 'gnn', 'aux_path': data_path_base + f'/graphs_sidelen_{highres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_val', 'type': 'pkl'}]
    if trainer_configs['gnn'].get('scales', 1) >= 2:
        other_aux.append({'name': 'gnn2', 'aux_path': data_path_base + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_train', 'type': 'pkl'})
        other_aux_test.append({'name': 'gnn2', 'aux_path': data_path_base + f'/graphs_sidelen_{lowres_graph_sidelen}_neg_0.1_v{graph_version}{graph_masked_str}{graph_psuncert_str}{graph_trained_num_str}_val', 'type': 'pkl'})
    wanted_size = tuple(full_config.get('image_size', [1024, 1024]))
    train_dataset = Dataset2d(data_path_train, mode='train', test_fd_str='TrMod', allow_no_label=True, only_no_label=True, last_stage_mask_path={'frangi': frangi_path}, other_aux=other_aux, load_mask_pseudo=data_path_base + '/PLPB_mask_train', norm_method=norm_method, wanted_size=wanted_size, augs=trainer_configs.get('augs', None))
    test_dataset = Dataset2d(data_path_test, mode='test', test_fd_str='TrMod', norm_method=norm_method, wanted_size=wanted_size, last_stage_mask_path={'frangi': frangi_path_test}, other_aux=other_aux_test)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=full_config.get('num_workers', 4), pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=full_config.get('num_workers', 4), pin_memory=True, collate_fn=custom_collate_fn)
    return (train_loader, test_loader)

def main():
    args = parse_args()
    full_config = load_yaml(args.config)
    configure_runtime(full_config)
    set_all_seeds(full_config.get('seed', 42))
    trainer_configs = full_config['trainer_configs']
    training_configs = full_config['training_configs']
    trainer_configs.update(training_configs)
    trainer_configs['training_base'] = 'tt_sfuda'
    trainer_configs['tta_use_model_as_teacher'] = not args.keep_config_teacher
    net_norm_method = full_config.get('net_norm_method', 'batch')
    if net_norm_method == 'instance':
        set_eval = True
    elif net_norm_method == 'batch':
        set_eval = False
    else:
        raise ValueError(f'Invalid norm layer {net_norm_method}')
    if trainer_configs.get('pseudo_label_original_model_eval', False):
        set_eval = True
    set_eval = True
    train_loader, test_loader = build_loaders(full_config)
    loader = test_loader if args.split == 'test' else train_loader
    run_name = full_config.get('run_name', 'vesselgda')
    base_folder = args.base_folder or osp.join('results', run_name + '_tta')
    if not osp.exists(base_folder):
        os.makedirs(base_folder)
    write_yaml(osp.join(base_folder, 'config.yaml'), full_config)
    device = torch.device(full_config.get('device', 'cuda:0'))
    patch_size_description = trainer_configs.get('patch_size_description')
    tta_net_conf = patch_size_description if patch_size_description is not None else 'FIVES'
    trainer_configs['tta_net_conf'] = tta_net_conf
    trainer_configs['tta_norm_layer'] = net_norm_method
    trainer_configs['tta_image_size'] = tuple(full_config.get('image_size', [1024, 1024]))
    model = make_net_2d(net_conf=tta_net_conf, norm_layer=net_norm_method, image_size=trainer_configs['tta_image_size'])
    if training_configs['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], betas=training_configs['optimizer_params']['betas'] if training_configs.get('optimizer_params') else (0.9, 0.999))
    elif training_configs['optimizer_type'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    elif training_configs['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'], momentum=0.9)
    elif training_configs['optimizer_type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=training_configs['lr'], weight_decay=training_configs['weight_decay'])
    else:
        raise ValueError(f"Invalid optimizer type {training_configs['optimizer_type']}")
    model = model.to(device)
    trainer = TrainerGNNTTA(model, optimizer, None, device, train_loader, test_loader, base_folder=base_folder, set_to_eval_when_testing=set_eval, **trainer_configs)
    if trainer_configs.get('trained_path'):
        trainer.load_models(trainer_configs['trained_path'])
    tta_gnn_path = None
    tta_refiner_path = None
    if trainer_configs.get('pretrain_gnn_path'):
        tta_gnn_path = trainer_configs['pretrain_gnn_path']
        tta_refiner_path = trainer_configs['pretrain_gnn_path'].replace('gnn', 'refiner')
        trainer.load_gnn(tta_gnn_path, tta_refiner_path)
    trainer.configure_tta_reinitialization(model_path=trainer_configs.get('trained_path'), gnn_path=tta_gnn_path, refiner_path=tta_refiner_path, use_model_as_teacher=trainer_configs['tta_use_model_as_teacher'])
    tta_steps = args.tta_steps if args.tta_steps is not None else trainer_configs.get('tta_steps', trainer_configs['max_epoch'])
    loader_name = args.loader_name
    if args.gnn_distillation_iters > 0:
        loader_name = f'{loader_name}_gnn_distill_{args.gnn_distillation_iters}'
    trainer.tta_test(loader, loader_name=loader_name, tta_steps=tta_steps, thres=args.thres, max_cases=args.max_cases, save_pred_interval=args.save_pred_interval, gnn_distillation_iters=args.gnn_distillation_iters, resume=not args.no_resume)
if __name__ == '__main__':
    main()
