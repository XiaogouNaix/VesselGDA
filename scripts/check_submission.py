"""Check imports, graph serialization, label isolation and topology refinement."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-3d', action='store_true')
    args = parser.parse_args()
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    modules = ['exp_gnn_pseudo_learn_2d', 'exp_gnn_pseudo_learn_2d_tta',
               'save_uncertainty_volume_2d', 'preprocess_build_graph_from_label_2d_goodfeature',
               'preprocess_frangi_filter_2d', 'test_combine_gnn_results_and_net_result']
    if args.with_3d:
        modules += ['exp_gnn_pseudo_learn', 'exp_gnn_pseudo_learn_tta']
    for name in modules:
        module = importlib.import_module(name)
        if ROOT not in Path(module.__file__).resolve().parents:
            raise AssertionError(f'Import escaped the release directory: {name}')

    import cv2
    import numpy as np
    import torch
    from dataset import Dataset2d
    from trainer import Trainer2d
    from test_combine_gnn_results_and_net_result import combine_predictions
    from preprocess_build_graph_from_label_2d_goodfeature import build_graph_with_uncertainty
    from seg_networks.graph_unet import TwoScaleGATInterp

    # A connected branch must survive; an isolated false positive must not.
    early = np.zeros((32, 32), dtype=np.uint8)
    early[8:18, 8:12] = 1
    early[1, 1] = 1
    late = np.zeros_like(early)
    late[12, 11:24] = 1
    late[25:28, 25:28] = 1
    trunk, final = combine_predictions(early, late, min_size=20)
    assert trunk.sum() == 40
    assert final[12, 23] == 1 and final[26, 26] == 0 and final[1, 1] == 0
    assert np.all(final[trunk > 0] == 1)
    assert not combine_predictions(np.zeros_like(early), late, 20)[1].any()
    diagonal = np.zeros_like(early)
    diagonal[18, 12] = 1
    assert combine_predictions(early, diagonal, 20)[1][18, 12] == 1

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        case = folder / 'train' / 'sample'
        case.mkdir(parents=True)
        cv2.imwrite(str(case / 'image.png'), np.full((32, 32, 3), 128, np.uint8))
        cv2.imwrite(str(case / 'label.png'), np.full((32, 32), 255, np.uint8))
        dataset = Dataset2d(str(case.parent), allow_no_label=True, only_no_label=True,
                            norm_method={'name': 'png_norm'}, augs=None)
        assert dataset[0]['label'].sum() == 0, 'Target labels entered the adaptation dataset.'
        # A constant foreground prediction must stay white after PNG serialization.
        dummy = type('SnapshotWriter', (), {'pred_path': str(folder)})()
        Trainer2d._do_save_image_val(dummy, {'name': ['sample']}, np.ones((32, 32)), 'Val_model', 0)
        saved = cv2.imread(str(folder / 'Val_model/epoch_0/sample_pred.png'), cv2.IMREAD_GRAYSCALE)
        assert np.all(saved == 255)

        class LabelFreeTrainer(Trainer2d):
            def train_epoch(self, epoch):
                with torch.no_grad():
                    self.model[0].bias.fill_(epoch + 1)

            def _wrap_model_forward_eval(self, **kwargs):
                return self.model

        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.BatchNorm2d(1), torch.nn.Sigmoid())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # The export loader intentionally has no label key.
        export_loader = [{'image': torch.ones(1, 1, 32, 32), 'name': ['sample']}]
        trainer = LabelFreeTrainer(model, optimizer, None, torch.device('cpu'), [], export_loader,
                                   max_epoch=2, base_folder=str(folder), patch_size=(32, 32),
                                   save_val_pred=True, save_val_epochs=[0, 1], set_to_eval_when_testing=False)
        buffers_before = {k: v.clone() for k, v in model.named_buffers()}
        trainer.train()
        assert all(torch.equal(v, buffers_before[k]) for k, v in model.named_buffers())
        final_state = torch.load(folder / 'state_dict_model_final.pt', map_location='cpu')
        assert final_state['0.bias'].item() == 2, 'Checkpoint did not retain the fixed final epoch.'
        assert (folder / 'Val_model/epoch_1/sample_pred.png').is_file() is False
        assert (folder / 'pred/Val_model/epoch_1/sample_pred.png').is_file()
        assert 'Val model:' not in (folder / 'log.txt').read_text()

    pseudo = np.zeros((64, 64), np.float32)
    pseudo[8:56, 28:36] = 1
    certainty = np.ones_like(pseudo)
    certainty[:, 20:44] = 0
    graphs = []
    for side in [16, 32]:
        gt, cert, coords, edges, weights = build_graph_with_uncertainty(
            pseudo, certainty, cell_size=(side, side), center_type='inscribed',
            new_find_center=True, certain_pseudo_only=False, tn_do_not_connect=2,
            image_mask=np.ones_like(pseudo), goodfeatures=None,
            centerline={'use': True, 'downsample': side // 8}, connect_to_tree_thres=20,
            max_edge_for_each_node=5)
        assert len(coords) == len(gt) == len(cert)
        assert edges.ndim == 2 and edges.shape[1] == 2 and len(edges) == len(weights)
        assert edges.min() >= 0 and edges.max() < len(coords)
        assert np.isfinite(weights).all() and np.all((coords >= 0) & (coords < 64))
        graphs.append((coords, edges, weights))
    teacher = TwoScaleGATInterp(4, 8, 1, num_layers=2, v2=False, heads=1,
                               feature_up=True, feature_down=True, density_correction='jackson',
                               twod=True, residual=True, norm=True)
    inputs = []
    for coords, edges, weights in graphs:
        inputs.extend([torch.randn(len(coords), 4, requires_grad=True),
                       torch.as_tensor(edges.T, dtype=torch.long),
                       torch.as_tensor(weights), torch.as_tensor(coords, dtype=torch.float32)])
    outputs = teacher(*inputs)
    assert all(torch.isfinite(pred).all() for pred in outputs)
    sum(pred.square().mean() for pred in outputs).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in teacher.parameters())
    print(json.dumps({'imports': len(modules), 'label_isolation': 'passed',
                      'graph_scales': [16, 32], 'gnn_backward': 'passed',
                      'snapshot_png': 'passed', 'merge': 'passed'}))


if __name__ == '__main__':
    main()
