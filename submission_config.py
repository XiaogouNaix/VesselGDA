"""Shared paths and validation for the standalone release."""
import os
from pathlib import Path


def configure_runtime(config):
    for key, variable in [('source_checkpoint', 'VESSELGDA_SOURCE_CHECKPOINT'),
                          ('source_model_dir', 'VESSELGDA_SOURCE_MODEL_DIR')]:
        if config.get(key):
            os.environ[variable] = str(config[key])


def dataset_root(config):
    return Path(config.get('data_root', f"AIF-SFDA-main/datasets/{config['dataset']}"))


def graph_folder(config, split, side):
    suffix = '_masked' if config.get('graph_masked', False) else ''
    if config.get('graph_psuncert', False):
        suffix += '_psuncert'
    if config.get('graph_trained_num') is not None:
        suffix += '_' + str(config['graph_trained_num'])
    return dataset_root(config) / f"graphs_sidelen_{side}_neg_0.1_v{config['graph_version']}{suffix}_{split}"


def cases(config, split):
    folder = dataset_root(config) / split
    paths = sorted(p for p in folder.iterdir() if p.is_dir())
    if not paths:
        raise ValueError(f'No cases found in {folder}')
    for path in paths:
        if not (path / 'image.png').is_file():
            raise FileNotFoundError(path / 'image.png')
    return paths


def validate_inputs(config):
    """Fail before adaptation if preprocessing outputs or selected epochs are missing."""
    import cv2
    import numpy as np
    import pickle
    size = tuple(config.get('image_size', [1024, 1024]))
    sides = [config.get('highres_graph_sidelen', 16)]
    if config['trainer_configs']['gnn'].get('scales', 1) >= 2:
        sides.append(config.get('lowres_graph_sidelen', 32))
    for split in ['train', 'val']:
        for case in cases(config, split):
            image = cv2.imread(str(case / 'image.png'))
            if image is None or image.shape[:2] != size:
                raise ValueError(f'{case}: expected an image of size {size}; standardize before preprocessing.')
            if split == 'val' and not (case / 'label.png').is_file():
                raise FileNotFoundError(case / 'label.png')
            for path in [dataset_root(config) / f'frangi_{split}' / (case.name + '.pt')]:
                with path.open('rb') as stream:
                    arr = pickle.load(stream)['arr']
                if arr.shape != size or not np.isfinite(arr).all():
                    raise ValueError(f'Invalid Frangi array: {path}')
            for side in sides:
                path = graph_folder(config, split, side) / (case.name + '.pkl')
                with path.open('rb') as stream:
                    graph = pickle.load(stream)
                if graph['coords'].ndim != 2 or graph['coords'].shape[1] != 2:
                    raise ValueError(f'Invalid graph coordinates: {path}')
                if graph['edges'].ndim != 2 or graph['edges'].shape[1] != 2 or not len(graph['edges']):
                    raise ValueError(f'Graph has no valid edges: {path}; inspect the source predictions and FOV.')
            if split == 'train':
                for kind in ['mask', 'pseudo']:
                    path = dataset_root(config) / 'PLPB_mask_train' / f'{case.name}_{kind}.pkl'
                    if not path.is_file():
                        raise FileNotFoundError(path)
    trainer = config['trainer_configs']
    for epoch in trainer.get('save_val_epochs', []) or []:
        if not 0 <= epoch < trainer['max_epoch']:
            raise ValueError(f'Snapshot epoch {epoch} must be in [0, max_epoch).')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Check prepared 2D inputs before adaptation.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    validate_inputs(config)
    print('All required inputs and snapshot epochs are valid.')
