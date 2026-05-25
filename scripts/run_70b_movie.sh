#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u run.py \
  --model_profile 70b \
  --dataset_name movie \
  --sample_size 200 \
  --gpu_ids 0,1 \
  --tensor_parallel_size 2 \
  --quantization awq
