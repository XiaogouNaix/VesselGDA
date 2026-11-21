"""Trunk-constrained topology refinement from saved dense CNN predictions.

The merge sequence and connectivity match the original experiment script.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_objects

from utils_preprocess import get_intersecting_components

# Fill with the directory of saved dense CNN masks used as the stable trunk reference.
# Expected filenames: <case_id>_pred.png, for example under pred/Val_model/epoch_<E>.
fd_pred = ""

# Fill with the directory of saved dense CNN masks after topology distillation.
# Use the same case IDs and image size as fd_pred. These are not GNN node visualizations.
fd_pred_with_gnn = ""


def combine_predictions(early, late, min_size=10000):
    if early.ndim != 2 or early.shape != late.shape:
        raise ValueError('Early and late masks must be 2D with the same shape.')
    # Original CCF uses skimage's default connectivity=1 (4-neighbour in 2D).
    trunk = remove_small_objects(early.astype(bool), min_size=min_size, connectivity=1)
    union = late.astype(bool) | trunk
    # Original intersection helper uses 8-neighbour components on the union.
    final = get_intersecting_components(trunk.astype(np.uint8), union.astype(np.uint8))
    return trunk.astype(np.uint8), final


def read_mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return (image > 127).astype(np.uint8)


def calc_results(pred_img, gt_img):
    from metrics import dice_np, clDice_np, precision_recall_np, iou_np, hd95_np, asd_np
    precision, recall = precision_recall_np(pred_img, gt_img)
    result = dict(dice=dice_np(pred_img, gt_img), cldice=clDice_np(pred_img, gt_img),
                  miou=iou_np(pred_img, gt_img), precision=precision, recall=recall)
    # Distances are undefined for empty boundaries; report null instead of crashing.
    result['hd95'] = hd95_np(pred_img, gt_img) if pred_img.any() and gt_img.any() else None
    result['asd'] = asd_np(pred_img, gt_img) if pred_img.any() and gt_img.any() else None
    return {k: float(v) if v is not None and np.isfinite(v) else None for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='YAML with merge settings; direct path options override it.')
    parser.add_argument('--early-dir', type=Path)
    parser.add_argument('--late-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gt-dir', type=Path, help='Optional evaluation labels: CASE_ID/label.png.')
    parser.add_argument('--min-size', type=int)
    args = parser.parse_args()
    merge = {}
    if args.config:
        import yaml
        with open(args.config) as stream:
            config = yaml.safe_load(stream)
        merge = config['merge']
        args.early_dir = args.early_dir or (Path(merge['early_dir']) if merge.get('early_dir') else None)
        args.late_dir = args.late_dir or (Path(merge['late_dir']) if merge.get('late_dir') else None)
        args.output_dir = args.output_dir or (Path(merge['output_dir']) if merge.get('output_dir') else None)
    args.early_dir = args.early_dir or (Path(fd_pred) if fd_pred else None)
    args.late_dir = args.late_dir or (Path(fd_pred_with_gnn) if fd_pred_with_gnn else None)
    if args.late_dir is not None and args.output_dir is None:
        args.output_dir = args.late_dir.with_name(args.late_dir.name + '_combined')
    if any(p is None for p in [args.early_dir, args.late_dir, args.output_dir]):
        parser.error('Fill fd_pred and fd_pred_with_gnn, set merge.early_dir and merge.late_dir in the YAML, or supply --early-dir and --late-dir.')
    min_size = args.min_size if args.min_size is not None else merge.get('min_size', 10000)
    if min_size < 0:
        parser.error('--min-size must be nonnegative.')
    if args.output_dir.resolve() in [args.early_dir.resolve(), args.late_dir.resolve()]:
        parser.error('Output must be separate from both input prediction directories.')
    early_paths = {p.name: p for p in args.early_dir.glob('*_pred.png')}
    late_paths = {p.name: p for p in args.late_dir.glob('*_pred.png')}
    if not early_paths or early_paths.keys() != late_paths.keys():
        raise ValueError(f'Nonempty, matching prediction sets required. Early-only: {sorted(early_paths.keys() - late_paths.keys())}; late-only: {sorted(late_paths.keys() - early_paths.keys())}')
    # Validate every input before writing any output.
    for name, path in early_paths.items():
        if read_mask(path).shape != read_mask(late_paths[name]).shape:
            raise ValueError(f'Mismatched prediction sizes: {name}')
        if args.gt_dir is not None and not (args.gt_dir / name[:-9] / 'label.png').is_file():
            raise FileNotFoundError(args.gt_dir / name[:-9] / 'label.png')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(early_paths):
        trunk, final = combine_predictions(read_mask(early_paths[name]), read_mask(late_paths[name]), min_size)
        if not cv2.imwrite(str(args.output_dir / name), final * 255):
            raise IOError(f'Cannot save {args.output_dir / name}')
        row = {'case': name[:-9]}
        if args.gt_dir:
            gt = cv2.imread(str(args.gt_dir / row['case'] / 'label.png'), cv2.IMREAD_GRAYSCALE)
            if gt.shape != final.shape:
                gt = cv2.resize(gt, final.shape[::-1], interpolation=cv2.INTER_LINEAR)
            gt = (gt > 127).astype(np.uint8)
            row.update(before=calc_results(trunk, gt), combined=calc_results(final, gt))
        rows.append(row)
    summary = dict(early_dir=str(args.early_dir), late_dir=str(args.late_dir), min_size=min_size,
                   ccf_connectivity=4, union_connectivity=8, cases=rows)
    if args.gt_dir:
        summary['mean'] = {}
        for stage in ['before', 'combined']:
            summary['mean'][stage] = {}
            for key in rows[0][stage]:
                values = [r[stage][key] for r in rows if r[stage][key] is not None]
                summary['mean'][stage][key] = float(np.mean(values)) if values else None
    (args.output_dir / 'merge_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Saved {len(rows)} combined predictions to {args.output_dir}')


if __name__ == '__main__':
    main()
