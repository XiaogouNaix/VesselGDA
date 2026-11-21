# VesselGDA

Code for *Hierarchical Structural Adaptation with Graph-Based Topology Reasoning for Source-Free Cross-Domain Vessel Segmentation*.

## Setup

Use Python 3.9. Run commands from this directory:

```bash
python -m pip install -r requirements.txt
# Additional dependencies for 3D:
python -m pip install -r requirements-3d.txt
```

Provide the pretrained FIVES source model separately as `checkpoints/fives_batch.pth`. It must be a state dictionary compatible with the included 2D backbone and configured normalization.

## Data

Example images illustrate the directory layout. Supply the complete dataset for experiments:

```text
AIF-SFDA-main/datasets/
  FIVES/imagesTr/<case_id>/image.png, label.png, mask.png
  DIAS/
    train/<case_id>/image.png, mask.png
    val/<case_id>/image.png, label.png, mask.png
  DSCA/
    train/<case_id>/image.png, mask.png
    val/<case_id>/image.png, label.png, mask.png
```

Target images must be 1024×1024; zero-pad non-square images before resizing. `mask.png` is the field of view, and `label.png` is the vessel annotation for final evaluation. Binary masks use 0/255. Grayscale DSA images are read as three channels. Adaptation does not require FIVES images.

## Run Fundus-to-DSA

Edit paths and parameters in `configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml` or `fundus_to_dsca.yaml`, then run:

```bash
python save_uncertainty_volume_2d.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
python preprocess_frangi_filter_2d.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
python preprocess_build_graph_from_label_2d_goodfeature.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
python submission_config.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
python exp_gnn_pseudo_learn_2d.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
```

Training uses the configured epoch count without intermediate test-label comparisons or checkpoint selection. It saves `state_dict_model_final.pt` and evaluates after training.

Set `save_val_epochs` to the zero-based epochs needed for merging (`null` saves all epochs). Exports compute no label-based metrics and are saved under `<output_dir>/pred/Val_model/epoch_<E>/`.

## Merge predictions

Fill the two empty paths `fd_pred` and `fd_pred_with_gnn` in `test_combine_gnn_results_and_net_result.py`, following the English comments, then run the script. Alternatively:

```bash
python test_combine_gnn_results_and_net_result.py \
  --early-dir PATH_TO_TRUNK_PREDICTIONS \
  --late-dir PATH_TO_POST_DISTILLATION_PREDICTIONS \
  --output-dir results/combined
```

Both directories must contain matching `<case_id>_pred.png` files. The script applies CCF to the trunk reference, unions it with the later prediction, and retains components intersecting the trunk. Add `--gt-dir AIF-SFDA-main/datasets/DIAS/val` for final evaluation. Merge paths can also be supplied through the YAML with `--config`.

## Other entry points

```bash
python exp_gnn_pseudo_learn_2d_tta.py --config configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml
python exp_gnn_pseudo_learn.py --config configs/exp_gnn_pseudo_learn/tof_to_snap.yaml
python exp_gnn_pseudo_learn_tta.py --config configs/exp_gnn_pseudo_learn/tof_to_snap.yaml
```

TTA resets state for each case and evaluates after its configured updates. 3D requires separately prepared volumes and a compatible nnU-Net source checkpoint with its plans and dataset metadata. The bundled end-to-end preprocessing workflow covers 2D Fundus-to-DSA.

Run `python scripts/check_submission.py` to check the installation (add `--with-3d` for 3D imports).
