#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u run.py --model_profile 8b --dataset_name ACE2005 --sample_size 200
