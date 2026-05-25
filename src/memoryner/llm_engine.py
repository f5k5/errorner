import os
from typing import Any, List, Optional


def load_embedding_model(model_path: str, device: str = "cuda"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, device=device)


def load_llm(runtime_config):
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = runtime_config.gpu_ids
    llm_kwargs = {
        "model": runtime_config.model_path,
        "tensor_parallel_size": runtime_config.tensor_parallel_size,
        "max_model_len": runtime_config.max_model_len,
        "gpu_memory_utilization": runtime_config.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }
    if runtime_config.quantization:
        llm_kwargs["quantization"] = runtime_config.quantization
    return LLM(**llm_kwargs)


def generate_texts(
    llm: Any,
    prompts: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    from vllm import SamplingParams

    return llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
    )


def get_token_stats(request_outputs) -> int:
    total = 0
    for request_output in request_outputs:
        total += len(getattr(request_output, "prompt_token_ids", []) or [])
        for output in getattr(request_output, "outputs", []) or []:
            total += len(getattr(output, "token_ids", []) or [])
    return total
