import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm
import torchio as tio
from metrics import dice_torch, precision_recall_torch, clDice_torch, hd95_torch, asd_torch, iou_torch
import os.path as osp
import os
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
import copy
import matplotlib.pyplot as plt
from utils import separate_to_batches, TxtLogger, dict_element_add, dict_element_divide, dict_element_lst_append, define_fd
from utils_preprocess import threshold_torch

def calc_metrics_torch(pred, target, full=True):
    dice = dice_torch(pred, target)
    precision, recall = precision_recall_torch(pred, target)
    iou = iou_torch(pred, target)
    cl_dice = clDice_torch(pred, target)
    if full:
        if pred.sum() == 0:
            hd95, asd = (1000.0, 1000.0)
        else:
            hd95 = hd95_torch(pred, target)
            asd = asd_torch(pred, target)
    else:
        hd95, asd = (1000.0, 1000.0)
    return {'dice': dice, 'precision': precision, 'recall': recall, 'iou': iou, 'cl_dice': cl_dice, 'hd95': hd95, 'asd': asd}

def weight_mask(D, H, W, w=0.5):
    weight = np.ones((D, H, W))
    weight[0:D // 6, :, :] = w
    weight[-D // 6:, :, :] = w
    weight[:, 0:H // 6, :] = w
    weight[:, -H // 6:, :] = w
    weight[:, :, 0:W // 6] = w
    weight[:, :, -W // 6:] = w
    return weight

class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class WrappedModelBase(nn.Module):

    def __init__(self, model):
        super(WrappedModelBase, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError()

class Trainer(nn.Module):

    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, max_epoch, base_folder, patches_in_batch=2, patch_size=(64, 64, 64), fg_oversample=0.5, set_to_eval_when_testing=True, webdataset=None, webdataset_val=None, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=max_epoch, eta_min=5e-07, last_epoch=-1)
        self.warmup_scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=2.0, warm_epoch=max_epoch // 10 + 1, after_scheduler=self.scheduler)
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.base_folder = base_folder
        self.log_path = osp.join(base_folder, 'log.txt')
        self.logger = TxtLogger(self.log_path, verbose=True)
        self.pred_path = osp.join(base_folder, 'pred')
        if not osp.exists(self.pred_path):
            os.makedirs(self.pred_path)
        self.model_path = {'model': osp.join(base_folder, 'state_dict_model_init.pt')}
        self.patches_in_batch = patches_in_batch
        self.patch_size = patch_size
        self.test_batch_size = 3
        self.set_to_eval_when_testing = set_to_eval_when_testing
        self.sampler = tio.sampler.LabelSampler(patch_size=patch_size, label_name='label', label_probabilities={0: 1 - fg_oversample, 1: fg_oversample})
        self.kwargs = kwargs
        self.orig_model = copy.deepcopy(self.model)
        self.teacher_model = copy.deepcopy(self.model)
        self.anchor_model = copy.deepcopy(self.orig_model)
        for param in self.orig_model.parameters():
            param.requires_grad = False
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        if kwargs.get('ema'):
            self.model_path['teacher'] = osp.join(base_folder, 'state_dict_teacher_model_init.pt')
            self.model_path['original'] = osp.join(base_folder, 'state_dict_original_model_init.pt')
        self.models_dict = {'model': self.model}
        self.models_state_dict_dict = {}
        if self.kwargs.get('ema'):
            self.models_dict['original'] = self.orig_model
        self.key_for_saving = 'model'
        self.aux_models_dict = {}
        self.webdataset = webdataset
        self.webdataset_val = webdataset_val
        self.use_webdataset = webdataset is not None
        if self.use_webdataset:
            self.train_loader = None
            self.test_loader = None

    def _normalize(self, x):
        return x

    def _init_models_dict(self, init_value):
        dic = {}
        for key in self.models_dict.keys():
            dic[key] = init_value
        return dic

    def _set_eval(self):
        if self.set_to_eval_when_testing:
            for model in self.models_dict.values():
                model.eval()
        else:
            for model in self.models_dict.values():
                model.train()

    def webdataset_get_loader(self):
        self.train_loader = self.webdataset.get_dataloader()
        self.test_loader = self.webdataset_val.get_dataloader()

    def train(self, val_interval=None, **kwargs):
        # Epoch count and snapshot schedule are fixed in the configuration.
        # Target labels are never used for stopping or checkpoint selection.
        for epoch in range(self.max_epoch):
            self.model.train()
            self.train_epoch(epoch, **kwargs)
            epochs = self.kwargs.get('save_val_epochs')
            if self.kwargs.get('save_val_pred', False) and (epochs is None or epoch in epochs):
                self.save_epoch_predictions(epoch)
            self.warmup_scheduler.step()
            self.logger.write('Main model lr: {:.6f}'.format(self.warmup_scheduler.after_scheduler.optimizer.param_groups[0]['lr']))
        for key, model in self.models_dict.items():
            path = osp.join(self.base_folder, f'state_dict_{key}_final.pt')
            torch.save(model.state_dict(), path)
            self.model_path[key] = path
        for key, model in self.aux_models_dict.items():
            folder = define_fd(osp.join(self.base_folder, key + '_models'))
            torch.save(model.state_dict(), osp.join(folder, f'state_dict_{key}_final.pt'))

    def train_epoch(self, epoch):
        total_loss = 0
        total_dice = 0
        idx = 0
        with tqdm(self.train_loader) as t:
            for batch in t:
                image = batch['image']
                label = batch['label']
                idx += label.shape[0]
                image = self._normalize(image)
                subject = tio.Subject(image=tio.ScalarImage(tensor=image.squeeze(0)), label=tio.LabelMap(tensor=label.squeeze(0)))
                samples = self.sampler(subject, self.patches_in_batch)
                image_lst = []
                label_lst = []
                for sample in samples:
                    image_lst.append(sample['image']['data'].unsqueeze(0))
                    label_lst.append(sample['label']['data'].unsqueeze(0))
                image = torch.cat(image_lst, dim=0).to(device=self.device, dtype=torch.float32)
                label = torch.cat(label_lst, dim=0).to(device=self.device, dtype=torch.float32)
                self.optimizer.zero_grad()
                pred = self.model(image)
                loss = self.criterion(pred, label)
                pred = threshold_torch(pred, 0.5)
                dice = dice_torch(pred, label)
                total_loss += loss.item()
                total_dice += dice * label.shape[0]
                loss.backward()
                self.optimizer.step()
                t.set_postfix(loss=total_loss / idx, dice=total_dice / idx, lr=self.optimizer.param_groups[0]['lr'])
        train_str = f'Epoch {epoch} - Train: Loss: {total_loss / idx:.4f}, Dice: {total_dice / idx:.4f}'
        self.logger.write(train_str)

    def save_epoch_predictions(self, epoch):
        # Export dense masks for later stage merging without reading target labels.
        # Preserve model modes and buffers so export cannot change adaptation state.
        for key, model in self.models_dict.items():
            modes = [(module, module.training) for module in model.modules()]
            buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
            try:
                self._set_eval()
                wrapped_model = self._wrap_model_forward_eval(model_to_test=key)
                with torch.no_grad():
                    for batch in self.val_loader:
                        image = self._normalize(batch['image']).to(device=self.device, dtype=torch.float32)
                        pred = sliding_window_inference(image, self.patch_size, self.test_batch_size, wrapped_model)
                        pred = self._postprocess(threshold_torch(pred.cpu().float(), 0.5))
                        self._do_save_image_val(batch, pred.squeeze(0).squeeze(0).numpy(), f'Val_{key}', epoch, postfix='_pred')
            finally:
                with torch.no_grad():
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                for module, training in modes:
                    module.training = training

    def test(self, loader, loader_name='Test', model_path=None, model_to_test='model', base_model_path=None, thres=0.5):
        self._set_eval()
        if base_model_path is not None:
            self.model.load_state_dict(torch.load(base_model_path, map_location=self.device), strict=False)
        if model_path is not None:
            self.models_dict[model_to_test].load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        total_metrics = {}
        idx = 0
        key_in_path = '' if model_to_test == 'model' else f'_{model_to_test}'
        loader_name = loader_name + key_in_path
        if not osp.exists(osp.join(self.pred_path, loader_name)):
            os.makedirs(osp.join(self.pred_path, loader_name))
        logger_new = TxtLogger(osp.join(self.pred_path, loader_name, f'results.txt'), verbose=True)
        with tqdm(loader) as t:
            for batch in t:
                image = batch['image']
                label = batch['label']
                idx += label.shape[0]
                image = self._normalize(image)
                image = image.to(device=self.device, dtype=torch.float32)
                wrapped_model = self._wrap_model_forward_eval(model_to_test=model_to_test)
                with torch.no_grad():
                    pred = sliding_window_inference(image, self.patch_size, self.test_batch_size, wrapped_model)
                pred = pred.cpu().to(dtype=torch.float32)
                pred = threshold_torch(pred, thres)
                pred = self._postprocess(pred)
                label = label.cpu().to(dtype=torch.float32)
                for i in range(pred.shape[0]):
                    pred_i = pred[i:i + 1]
                    label_i = label[i:i + 1]
                    if pred_i.shape != label_i.shape:
                        raise ValueError(f'Prediction shape {pred_i.shape} does not match label shape {label_i.shape}')
                    dice = dice_torch(pred_i, label_i)
                    metrics = calc_metrics_torch(pred_i, label_i)
                    logger_new.write(f"Dice for {batch['name'][i]}: {dice:.4f}. Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    total_metrics = dict_element_lst_append(total_metrics, metrics)
                pred = pred.squeeze(0).squeeze(0).numpy()
                self._do_save_image(batch, pred, loader_name, postfix='_pred')
        total_metrics_mean = {}
        total_metrics_std = {}
        for key, value in total_metrics.items():
            if isinstance(value, list):
                total_metrics_mean[key] = np.mean(value)
                total_metrics_std[key] = np.std(value)
            else:
                total_metrics_mean[key] = value
                total_metrics_std[key] = 0.0
        test_str = f'Test result: '
        for key, value in total_metrics_mean.items():
            test_str += f'{key}: {value:.4f} +- {total_metrics_std[key]:.4f}, '
        logger_new.write(test_str)
        self.logger.write(test_str)

    def _do_save_image(self, batch, pred, loader_name, postfix='_pred'):
        spacing = batch['spacing']
        spacing = (spacing[0].item(), spacing[1].item(), spacing[2].item())
        pred_image = sitk.GetImageFromArray(pred)
        pred_image.SetSpacing(spacing)
        sitk.WriteImage(pred_image, osp.join(self.pred_path, loader_name, batch['name'][0] + f'{postfix}.nii.gz'))

    def _do_save_image_val(self, batch, pred, loader_name, epoch, postfix='_pred'):
        spacing = batch['spacing']
        spacing = (spacing[0].item(), spacing[1].item(), spacing[2].item())
        pred_image = sitk.GetImageFromArray(pred)
        pred_image.SetSpacing(spacing)
        if not osp.exists(osp.join(self.pred_path, loader_name, f'epoch_{epoch}')):
            os.makedirs(osp.join(self.pred_path, loader_name, f'epoch_{epoch}'))
        sitk.WriteImage(pred_image, osp.join(self.pred_path, loader_name, f'epoch_{epoch}', batch['name'][0] + f'{postfix}.nii.gz'))

    def print_test_image(self, loader, loader_name='Test', model_path=None, postfix='_pred', thres=0.5):
        self._set_eval()
        if model_path is not None:
            print(f'Loading model from {model_path}')
            self.model.load_state_dict(torch.load(osp.join(self.base_folder, model_path)), strict=False)
        else:
            print('No model loaded')
            pass
        idx = 0
        if not osp.exists(osp.join(self.pred_path, loader_name)):
            os.makedirs(osp.join(self.pred_path, loader_name))
        with tqdm(loader) as t:
            for batch in t:
                idx += batch['image'].shape[0]
                image = batch['image']
                image = self._normalize(image)
                image = image.to(device=self.device, dtype=torch.float32)
                wrapped_model = self._wrap_model_forward_eval()
                with torch.no_grad():
                    pred = sliding_window_inference(image, self.patch_size, self.test_batch_size, wrapped_model)
                pred = pred.cpu().to(dtype=torch.float32)
                pred = threshold_torch(pred, thres)
                pred = self._postprocess(pred)
                pred = pred.squeeze(0).squeeze(0).numpy()
                self._do_save_image(batch, pred, loader_name, postfix=postfix)

    def _do_eval_volume(self, image, label, mode='average', wt_lower_bound=0.5, sigma=1.0, batch_size=12):
        flag = 0
        z_size = self.patch_size[0]
        y_size = self.patch_size[1]
        x_size = self.patch_size[2]
        if image.shape[1] < z_size:
            flag = 1
            diff = z_size - image.shape[1]
            image = F.pad(image, [0, 0, 0, 0, diff // 2, diff - diff // 2])
            label = F.pad(label, [0, 0, 0, 0, diff // 2, diff - diff // 2])
        image_test_list = []
        label_test_list = []
        test_cube_slices = z_size
        test_slices_interval = z_size // 2
        y_interval = y_size // 2
        x_interval = x_size // 2
        times = np.zeros(np.array(image.shape[1:4]))
        predf = np.zeros(np.array(image.shape[1:4]))
        assert image.shape[0] == 1
        inds = range(0, image.shape[1] - test_cube_slices, test_slices_interval)
        z_begins_lst = [i for i in inds]
        z_begins_lst.append(image.shape[1] - test_cube_slices)
        for ind in inds:
            image_test_list.append(image[:, ind:ind + test_cube_slices, :, :])
            label_test_list.append(label[:, ind:ind + test_cube_slices, :, :])
        image_test_list.append(image[:, image.shape[1] - test_cube_slices:image.shape[1], :, :])
        label_test_list.append(label[:, image.shape[1] - test_cube_slices:image.shape[1], :, :])
        if mode == 'average':
            W = weight_mask(test_cube_slices, y_size, x_size, w=wt_lower_bound)
        elif mode == 'max':
            W = np.ones((test_cube_slices, y_size, x_size))
        for num, (i, l, z) in enumerate(zip(image_test_list, label_test_list, z_begins_lst)):
            i = i[0]
            l = l[0]
            inds1 = range(0, i.shape[1] - y_size, y_interval)
            inds2 = range(0, i.shape[2] - x_size, x_interval)
            begins_lst_y = [idx for idx in inds1]
            begins_lst_x = [idx for idx in inds2]
            begins_lst_y.append(i.shape[1] - y_size)
            begins_lst_x.append(i.shape[2] - x_size)
            patch_lst = []
            label_lst = []
            begin_coord_lst = []
            for iy in begins_lst_y:
                for ix in begins_lst_x:
                    patch_lst.append(i[:, iy:iy + y_size, ix:ix + x_size])
                    label_lst.append(l[:, iy:iy + y_size, ix:ix + x_size])
                    begin_coord_lst.append((iy, ix))
            i = torch.stack(patch_lst)
            l = torch.stack(label_lst)
            sh = i.shape
            i = i.reshape([sh[0], 1, sh[1], sh[2], sh[3]])
            l = l.reshape([sh[0], 1, sh[1], sh[2], sh[3]])
            batch_lst = separate_to_batches(i, batch_size)
            batch_label_lst = separate_to_batches(l, batch_size)
            p_lst = []
            for i1, l1 in zip(batch_lst, batch_label_lst):
                i1 = i1.to(device=self.device, dtype=torch.float32)
                l1 = l1.to(device=self.device, dtype=torch.float32)
                pred = self._do_network_forward_eval(i1)
                pred = pred.data.cpu()
                p_lst.append(pred)
            p = torch.cat(p_lst, dim=0)
            p = np.array(p.data.cpu())
            p = np.squeeze(p)
            idx = 0
            for patch in p:
                x_coord = begin_coord_lst[idx][1]
                y_coord = begin_coord_lst[idx][0]
                predf[z:z + test_cube_slices, y_coord:y_coord + y_size, x_coord:x_coord + x_size] += patch * W
                times[z:z + test_cube_slices, y_coord:y_coord + y_size, x_coord:x_coord + x_size] += W
                idx += 1
        if mode == 'average':
            predf = predf / times
        elif mode == 'max':
            pass
        else:
            raise ValueError('mode can only be average or max')
        thres = 0.5
        predf[predf >= thres] = 1
        predf[predf < thres] = 0
        label = np.array(label[0])
        icube = np.array(image[0])
        if flag == 1:
            predf = predf[diff // 2:predf.shape[0] - diff + diff // 2, ...]
            label = label[diff // 2:label.shape[0] - diff + diff // 2, ...]
            icube = icube[diff // 2:icube.shape[0] - diff + diff // 2, ...]
        return predf

    def _do_network_forward_eval(self, input):
        return self.model(input)

    def _wrap_model_forward_eval(self, **kwargs):

        class WrappedModel(WrappedModelBase):

            def forward(self, x):
                return self.model(x)
        if kwargs.get('model_to_test'):
            return WrappedModel(self.models_dict[kwargs['model_to_test']])
        else:
            return WrappedModel(self.model)

    def _postprocess(self, pred):
        return pred


class Trainer2d(Trainer):

    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, max_epoch, base_folder, patches_in_batch=2, patch_size=(64, 64), fg_oversample=0.5, **kwargs):
        super(Trainer2d, self).__init__(model, optimizer, criterion, device, train_loader, val_loader, max_epoch, base_folder, patches_in_batch=patches_in_batch, patch_size=patch_size, fg_oversample=fg_oversample, **kwargs)

    def _do_save_image(self, batch, pred, loader_name, postfix='_pred'):
        cv2.imwrite(osp.join(self.pred_path, loader_name, batch['name'][0] + f'{postfix}.png'), (pred > 0.5).astype(np.uint8) * 255)

    def _do_save_image_val(self, batch, pred, loader_name, epoch, postfix='_pred'):
        if not osp.exists(osp.join(self.pred_path, loader_name, f'epoch_{epoch}')):
            os.makedirs(osp.join(self.pred_path, loader_name, f'epoch_{epoch}'))
        cv2.imwrite(osp.join(self.pred_path, loader_name, f'epoch_{epoch}', batch['name'][0] + f'{postfix}.png'), (pred > 0.5).astype(np.uint8) * 255)
