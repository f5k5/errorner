from dataclasses import dataclass
from typing import Dict, Optional

DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "movie": {
        "test": "",
        "train": "",
        "output": "",
        "domain": "",
    },
    "restaurant": {
        "test": "",
        "train": "",
        "output": "",
        "domain": "",
    },
    "GENIA": {
        "test": "",
        "train": "",
        "output": "",
        "domain": "",
    },
    "CoNLL2003": {
        "test": "",
        "train": "",
        "output": "",
        "domain": "",
    },
    "ACE2005": {
        "test": "",
        "train": "",
        "output": "",
        "domain": "",
    },
}

MODEL_PROFILES: Dict[str, Dict[str, object]] = {
    "8b": {
        "model_path": "",
        "model_name": "",
        "gpu_ids": "0",
        "tensor_parallel_size": 1,
        "quantization": None,
        "batch_size": 32,
    },
    "70b": {
        "model_path": "",
        "model_name": "",
        "gpu_ids": "0,1",
        "tensor_parallel_size": 2,
        "quantization": "awq",
        "batch_size": 16,
    },
}

DEFAULT_EMBEDDING_MODEL_PATH = ""
DEFAULT_OUTPUT_ROOT = ""

DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_VERIFY_MAX_TOKENS = 1024
DEFAULT_GLOBAL_SAMPLE_SIZE = 200
DEFAULT_K_NEIGHBORS = 20
DEFAULT_K_GLOBAL = 3


@dataclass
class RuntimeConfig:
    model_profile: str
    model_path: str
    model_name: str
    embedding_model_path: str
    gpu_ids: str
    tensor_parallel_size: int
    quantization: Optional[str]
    batch_size: int
    max_model_len: int
    gpu_memory_utilization: float
    temperature: float
    top_p: float
    max_tokens: int
    verify_max_tokens: int


def build_runtime_config(args) -> RuntimeConfig:
    profile = MODEL_PROFILES.get(args.model_profile, MODEL_PROFILES["8b"]).copy()
    return RuntimeConfig(
        model_profile=args.model_profile,
        model_path=args.model_path or str(profile["model_path"]),
        model_name=args.model_name or str(profile["model_name"]),
        embedding_model_path=args.embedding_model_path,
        gpu_ids=args.gpu_ids or str(profile["gpu_ids"]),
        tensor_parallel_size=args.tensor_parallel_size or int(profile["tensor_parallel_size"]),
        quantization=args.quantization if args.quantization is not None else profile["quantization"],
        batch_size=args.batch_size or int(profile["batch_size"]),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        verify_max_tokens=args.verify_max_tokens,
    )
