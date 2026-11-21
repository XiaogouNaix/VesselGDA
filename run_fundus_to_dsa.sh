#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
config="${1:-configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml}"
python_bin="${PYTHON_BIN:-python}"
"$python_bin" save_uncertainty_volume_2d.py --config "$config"
"$python_bin" preprocess_frangi_filter_2d.py --config "$config"
"$python_bin" preprocess_build_graph_from_label_2d_goodfeature.py --config "$config"
"$python_bin" submission_config.py --config "$config"
"$python_bin" exp_gnn_pseudo_learn_2d.py --config "$config"
"$python_bin" test_combine_gnn_results_and_net_result.py --config "$config"
