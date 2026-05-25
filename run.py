import argparse
import os
import random
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from memoryner.config import (
    DATASET_CONFIGS,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_GLOBAL_SAMPLE_SIZE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_K_GLOBAL,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_VERIFY_MAX_TOKENS,
    build_runtime_config,
)
from memoryner.experiment import run_one_dataset, write_cross_dataset_reports
from memoryner.llm_engine import load_embedding_model, load_llm
from memoryner.utils.text import color_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_profile", type=str, default="8b", choices=["8b", "70b"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--embedding_model_path", type=str, default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "awq", "gptq", "fp8"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu_memory_utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--verify_max_tokens", type=int, default=DEFAULT_VERIFY_MAX_TOKENS)

    parser.add_argument("--dataset_name", type=str, default="all", choices=["all", *DATASET_CONFIGS.keys()])
    parser.add_argument("--datasets", type=str, default="movie,restaurant,GENIA,CoNLL2003,ACE2005")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--sample_size", type=int, default=DEFAULT_GLOBAL_SAMPLE_SIZE)
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--k_global", type=int, default=DEFAULT_K_GLOBAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = build_runtime_config(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = runtime_config.gpu_ids

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(color_text("Runtime configuration", "cyan"))
    print(f"Model profile: {runtime_config.model_profile}")
    print(f"Model path: {runtime_config.model_path}")
    print(f"Embedding model path: {runtime_config.embedding_model_path}")
    print(f"GPU IDs: {runtime_config.gpu_ids}")
    print(f"Tensor parallel size: {runtime_config.tensor_parallel_size}")
    print(f"Quantization: {runtime_config.quantization}")

    print(color_text("Loading embedding model...", "cyan"))
    emb_model = load_embedding_model(runtime_config.embedding_model_path, device="cuda")

    print(color_text("Loading LLM with vLLM...", "cyan"))
    llm = load_llm(runtime_config)

    if args.dataset_name != "all":
        dataset_names = [args.dataset_name]
    else:
        dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    all_summaries = {}
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_CONFIGS:
            print(color_text(f"[Skip] Unknown dataset: {dataset_name}", "red"))
            continue
        all_summaries[dataset_name] = run_one_dataset(dataset_name, llm, emb_model, args, runtime_config)

    write_cross_dataset_reports(args.output_root, all_summaries)


if __name__ == "__main__":
    main()
