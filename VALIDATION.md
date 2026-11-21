# Release verification

## Checks completed during packaging

- Imported all four requested SFUDA/TTA entry points from the standalone release directory, without the original project on the import path.
- Confirmed that required definitions replacing the private utility package match the corresponding local utility and metric implementations before extraction.
- Compared syntax trees of the core 2D training step, graph-supervision loss, Frangi/pseudo refinement, graph construction, two-scale GAT, 2D local contrastive module and both uncertainty-voting functions against the original code. Their executable implementations were unchanged.
- Ran source-model pseudo-label/uncertainty generation, Frangi extraction and both graph-construction scales on one train case and one held-out case resized to 128×128 in a temporary test directory.
- Ran the actual 2D SFUDA entry point for two CPU epochs using the compatible FIVES source checkpoint. The reduced test enabled local consistency/contrastive learning from the first epoch and topology distillation in the second. Logged losses were finite, including graph and distillation losses.
- Exported both dense epoch predictions and ran the merge command with optional evaluation labels. The final PNG and JSON report were produced.
- Ran the 2D TTA entry point on one reduced-size case, including one dense adaptation update and one additional topology-distillation update, with saved iteration predictions and a final metric report.
- Checked graph index/weight validity, target-label exclusion from the adaptation loader, all-foreground PNG serialization, connected-branch retention, isolated-component rejection and empty-trunk behavior.

No trained weights, smoke-test outputs or temporary source checkpoint paths are included in the release. `scripts/check_submission.py` repeats the import and contract checks without a source checkpoint.

## Scope

The reduced test used a smaller spatial resolution, two epochs and a correspondingly smaller CCF threshold to exercise the pipeline. It is not a performance experiment. Full-resolution training, complete-dataset metric reproduction and 3D end-to-end execution were not performed during packaging. GPU execution was unavailable in the packaging session; CPU execution was used for the reduced test.

The recorded environment had an incompatible optional `torch-scatter` binary; PyTorch Geometric disabled that extension and the tested GAT path ran with its available fallback. Install optional compiled extensions only when compatible with the chosen PyTorch build.

## Changes made for the standalone release

- Preserved the requested entry-point filenames and dependency directory structure; extracted dependency definitions instead of copying complete experimental repositories.
- Replaced machine-specific paths with configuration values and removed the email exception-monitor dependency.
- Removed out-of-scope discriminator and AIF baseline execution branches from the 3D entry point, with explicit errors if requested.
- Restricted the graph-architecture selector to the published two-scale GATInterp teacher; unrelated graph-model alternatives and standalone dependency demos are excluded.
- Replaced the hardcoded preprocessing drivers with configurable drivers producing the exact directory names consumed by the 2D loader. Graph construction and the topology merge retain the original numerical operations.
- Corrected the pseudo generator's reference to a global device, and serialize generated tensors on CPU.
- Made `Dataset2d.only_no_label` effective. The original adaptation loop already discarded training labels before computing its losses; the release also avoids reading those labels.
- Added configured epoch exports with no target-label metrics. Snapshot inference restores model buffers and modes afterward. Training saves the fixed final checkpoint without test-Dice selection.
- Removed per-epoch graph/segmentor test comparisons and intermediate TTA Dice curves; evaluation runs after the configured adaptation duration.
- Used explicit binary PNG encoding, including the all-foreground edge case.
- Removed internal run identifiers. The two merge-input variables remain empty with English comments; YAML merge inputs are likewise unset.
- Made the 3D source model directory configurable and removed unused source-dataset loader construction. The retained source-free workflow uses only the supplied source checkpoint and target data.

The supplied configurations retain the implementation's optimization settings. Snapshot choices are intentionally left to the experiment protocol. This verification does not establish that those settings, every manuscript hyperparameter and every reported result have been reconciled.
