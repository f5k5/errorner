import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memoryner.llm_engine import generate_texts, get_token_stats
from memoryner.memory.formatting import format_global_coupled_memory, format_local_coupled_memory
from memoryner.metrics import compute_micro_f1
from memoryner.prompts import CRITIQUE_AGENT_PROMPT, LOCAL_REFLECTION_PROMPT, PREDICTION_AGENT_PROMPT, VERIFICATION_AGENT_PROMPT, ZERO_SHOT_PREDICTION_PROMPT
from memoryner.utils.io import save_json
from memoryner.utils.text import color_text, md5_text, normalize_prediction, parse_json_robust, safe_name


class MemoryNERAblationRunner:
    def __init__(
        self,
        llm: Any,
        emb_model: Any,
        train_data: List[Dict[str, Any]],
        labels_str: str,
        allowed_labels: List[str],
        domain_name: str,
        k_neighbors: int,
        k_global: int,
        batch_size: int,
        max_tokens: int,
        verify_max_tokens: int,
        validated_retriever: Optional[Any] = None,
        unvalidated_retriever: Optional[Any] = None,
    ):
        self.llm = llm
        self.emb_model = emb_model
        self.train_data = train_data
        self.labels_str = labels_str
        self.allowed_labels = allowed_labels
        self.domain_name = domain_name
        self.k_neighbors = k_neighbors
        self.k_global = k_global
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.verify_max_tokens = verify_max_tokens
        self.validated_retriever = validated_retriever
        self.unvalidated_retriever = unvalidated_retriever
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.local_token_usage = 0
        self.init_instruction = f"Allowed entity types: {self.labels_str}"

        print(color_text("Building embedding vector store...", "cyan"))
        self.train_texts = [item["text"] for item in train_data]
        self.train_embeddings = self.emb_model.encode(
            self.train_texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.variants = {
            "vanilla": {"mode": "zero_shot", "LS": False, "LE": False, "GS": False, "GE": False, "global_source": "validated", "local_source": "validated"},
            "rag": {"mode": "paper", "LS": False, "LE": True, "GS": False, "GE": False, "global_source": "validated", "local_source": "validated"},
            "MemoryNER": {"mode": "paper", "LS": True, "LE": True, "GS": True, "GE": True, "global_source": "validated", "local_source": "validated"},
            "w/o hierarchical rules": {"mode": "paper_flat", "LS": True, "LE": True, "GS": True, "GE": True, "global_source": "validated", "local_source": "validated"},
            "w/o Validation": {"mode": "paper", "LS": True, "LE": True, "GS": True, "GE": True, "global_source": "unvalidated", "local_source": "unvalidated"},
            "w/o global generic Memory": {"mode": "paper", "LS": True, "LE": True, "GS": True, "GE": False, "global_source": "validated", "local_source": "validated"},
            "w/o global Semantic Memory": {"mode": "paper", "LS": True, "LE": True, "GS": False, "GE": True, "global_source": "validated", "local_source": "validated"},
            "w/o local Semantic Memory": {"mode": "paper", "LS": False, "LE": True, "GS": True, "GE": True, "global_source": "validated", "local_source": "validated"},
            "w/o local generic Memory": {"mode": "paper", "LS": True, "LE": False, "GS": True, "GE": True, "global_source": "validated", "local_source": "validated"},
            "w/o Coupling": {"mode": "paper_no_coupling", "LS": True, "LE": True, "GS": True, "GE": True, "global_source": "validated", "local_source": "validated"},
        }

    def _retrieve_global_rules(self, test_texts: List[str]):
        validated_batch = [[] for _ in test_texts]
        unvalidated_batch = [[] for _ in test_texts]
        if self.validated_retriever:
            validated_batch = self.validated_retriever.retrieve(test_texts, top_k=self.k_global)
        if self.unvalidated_retriever:
            unvalidated_batch = self.unvalidated_retriever.retrieve(test_texts, top_k=self.k_global)
        return validated_batch, unvalidated_batch

    def _build_local_memories_for_neighbors(self, missing_indices: List[int]) -> None:
        if not missing_indices:
            return
        print(color_text(f"[Local] Building local memory for {len(missing_indices)} samples...", "cyan"))
        pred_prompts = [
            ZERO_SHOT_PREDICTION_PROMPT.format(init_instruction=self.init_instruction, text=self.train_data[idx]["text"])
            for idx in missing_indices
        ]
        pred_outputs = generate_texts(self.llm, pred_prompts, max_tokens=1024, temperature=0.0)
        self.local_token_usage += get_token_stats(pred_outputs)

        critique_prompts = []
        temp_preds = []
        for i, output in enumerate(pred_outputs):
            idx = missing_indices[i]
            item = self.train_data[idx]
            pred = normalize_prediction(parse_json_robust(output.outputs[0].text), self.allowed_labels, original_input_text=item["text"])
            temp_preds.append(pred)
            critique_prompts.append(
                CRITIQUE_AGENT_PROMPT.format(
                    text=item["text"],
                    ground_truth=json.dumps(item.get("entities", []), ensure_ascii=False),
                    prediction=json.dumps(pred, ensure_ascii=False),
                )
            )

        critique_outputs = generate_texts(self.llm, critique_prompts, max_tokens=512, temperature=0.0)
        self.local_token_usage += get_token_stats(critique_outputs)

        for i, output in enumerate(critique_outputs):
            idx = missing_indices[i]
            parsed = parse_json_robust(output.outputs[0].text)
            critique_text = parsed.get("critique", "Correct.") if isinstance(parsed, dict) else "Correct."
            self.memory[md5_text(self.train_data[idx]["text"])] = {
                "pred": temp_preds[i],
                "critique": critique_text,
                "local_rules_validated": [],
                "local_rules_unvalidated": [],
            }

        failed_indices = [
            idx
            for idx in missing_indices
            if "Correct" not in self.memory[md5_text(self.train_data[idx]["text"])] ["critique"]
        ]
        if not failed_indices:
            return

        print(color_text(f"[Local] Synthesizing local rules for {len(failed_indices)} failed samples...", "cyan"))
        local_prompts = []
        for idx in failed_indices:
            critique = self.memory[md5_text(self.train_data[idx]["text"])] ["critique"]
            local_prompts.append(LOCAL_REFLECTION_PROMPT.format(critiques_summary=f"- {critique}"))

        local_outputs = generate_texts(self.llm, local_prompts, max_tokens=256, temperature=0.1)
        self.local_token_usage += get_token_stats(local_outputs)

        validation_prompts = []
        validation_meta = []
        for i, output in enumerate(local_outputs):
            idx = failed_indices[i]
            item = self.train_data[idx]
            key = md5_text(item["text"])
            parsed = parse_json_robust(output.outputs[0].text)
            if isinstance(parsed, dict):
                rule_list = parsed.get("rules", [])
            elif isinstance(parsed, list):
                rule_list = parsed
            else:
                rule_list = []
            clean_rules = [rule.strip() for rule in rule_list if isinstance(rule, str) and rule.strip()]
            self.memory[key]["local_rules_unvalidated"].extend(clean_rules)
            for rule in clean_rules:
                validation_prompts.append(
                    VERIFICATION_AGENT_PROMPT.format(
                        entity_type_definitions=self.labels_str,
                        r=rule,
                        X_i=item["text"],
                    )
                )
                validation_meta.append({"idx": idx, "rule": rule})

        if not validation_prompts:
            return

        validation_outputs = generate_texts(self.llm, validation_prompts, max_tokens=self.verify_max_tokens, temperature=0.0)
        self.local_token_usage += get_token_stats(validation_outputs)

        for i, output in enumerate(validation_outputs):
            meta = validation_meta[i]
            idx = meta["idx"]
            item = self.train_data[idx]
            key = md5_text(item["text"])
            new_pred = normalize_prediction(parse_json_robust(output.outputs[0].text), self.allowed_labels, original_input_text=item["text"])
            gold = normalize_prediction(item.get("entities", []), self.allowed_labels)
            if sorted((e["entity_text"], e["entity_type"]) for e in new_pred) == sorted((e["entity_text"], e["entity_type"]) for e in gold):
                self.memory[key]["local_rules_validated"].append(meta["rule"])

    def _prepare_memories(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(color_text("\n[Memory] Preparing memories...", "cyan"))
        test_texts = [item["text"] for item in test_data]
        test_embeddings = self.emb_model.encode(test_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        sim_matrix = cosine_similarity(test_embeddings, self.train_embeddings)
        neighbors_indices = [np.argsort(sim_matrix[i])[-self.k_neighbors:][::-1].tolist() for i in range(len(test_data))]
        validated_rules_batch, unvalidated_rules_batch = self._retrieve_global_rules(test_texts)

        unique_indices = {idx for group in neighbors_indices for idx in group}
        missing_indices = [idx for idx in unique_indices if md5_text(self.train_data[idx]["text"]) not in self.memory]
        self._build_local_memories_for_neighbors(missing_indices)

        prepared = []
        for i, item in enumerate(test_data):
            memory_item = {
                "GS_RULES": [],
                "GE_CASES": [],
                "GSU_RULES": [],
                "GEU_CASES": [],
                "LS_RULES": [],
                "LSU_RULES": [],
                "LE_TRAJECTORIES": [],
                "RAG_TRAJECTORIES": [],
                "RAW_VALIDATED_RULES": validated_rules_batch[i],
                "RAW_UNVALIDATED_RULES": unvalidated_rules_batch[i],
            }

            for rule in validated_rules_batch[i]:
                memory_item["GS_RULES"].append(rule.get("rule", ""))
                for context in rule.get("contexts", []):
                    memory_item["GE_CASES"].append(
                        f"Text: {context.get('text', '')}\nGold Output: {json.dumps(context.get('gold', []), ensure_ascii=False)}"
                    )

            for rule in unvalidated_rules_batch[i]:
                memory_item["GSU_RULES"].append(rule.get("rule", ""))
                for context in rule.get("contexts", []):
                    memory_item["GEU_CASES"].append(
                        f"Text: {context.get('text', '')}\nGold Output: {json.dumps(context.get('gold', []), ensure_ascii=False)}"
                    )

            for rank, idx in enumerate(neighbors_indices[i], start=1):
                neighbor = self.train_data[idx]
                key = md5_text(neighbor["text"])
                local_memory = self.memory.get(key, {})
                gold = neighbor.get("entities", [])
                text_short = neighbor["text"][:300] + "..." if len(neighbor["text"]) > 300 else neighbor["text"]
                rag_case = (
                    f"[Retrieved Example {rank}]\n"
                    f"Example Text: {text_short}\n"
                    f"Example Output: {json.dumps(gold, ensure_ascii=False)}"
                )
                memory_item["RAG_TRAJECTORIES"].append(rag_case)

                for rule in local_memory.get("local_rules_validated", []):
                    if rule not in memory_item["LS_RULES"]:
                        memory_item["LS_RULES"].append(rule)
                for rule in local_memory.get("local_rules_unvalidated", []):
                    if rule not in memory_item["LSU_RULES"]:
                        memory_item["LSU_RULES"].append(rule)

                critique = local_memory.get("critique", "Correct.")
                critique_short = critique[:100] + "..." if len(critique) > 100 else critique
                if "Correct" in critique:
                    trajectory = (
                        f"[Reference Trajectory {rank}]\n"
                        f"Example Text: {text_short}\n"
                        f"Example Output: {json.dumps(gold, ensure_ascii=False)}"
                    )
                else:
                    trajectory = (
                        f"[Error Trajectory {rank}]\n"
                        f"Example Text: {text_short}\n"
                        f"Bad Prediction: {json.dumps(local_memory.get('pred', []), ensure_ascii=False)}\n"
                        f"Example Output: {json.dumps(gold, ensure_ascii=False)}\n"
                        f"Correction Note: {critique_short}"
                    )
                memory_item["LE_TRAJECTORIES"].append(trajectory)

            prepared.append(memory_item)
        return prepared

    def _build_prompt(self, item: Dict[str, Any], memory: Dict[str, Any], cfg: Dict[str, Any]) -> str:
        if cfg["mode"] == "zero_shot":
            return ZERO_SHOT_PREDICTION_PROMPT.format(init_instruction=self.init_instruction, text=item["text"])

        if cfg["global_source"] == "unvalidated":
            raw_global_rules = memory["RAW_UNVALIDATED_RULES"]
            gs_rules = memory["GSU_RULES"]
            ge_cases = memory["GEU_CASES"]
        else:
            raw_global_rules = memory["RAW_VALIDATED_RULES"]
            gs_rules = memory["GS_RULES"]
            ge_cases = memory["GE_CASES"]

        local_rules = memory["LSU_RULES"] if cfg["local_source"] == "unvalidated" else memory["LS_RULES"]
        local_trajectories = memory["RAG_TRAJECTORIES"] if cfg["mode"] == "paper" and not cfg["LS"] and cfg["LE"] else memory["LE_TRAJECTORIES"]

        if not cfg["LS"]:
            local_rules = []
        if not cfg["LE"]:
            local_trajectories = []
        if not cfg["GS"]:
            raw_global_rules = []
            gs_rules = []
        if not cfg["GE"]:
            if raw_global_rules:
                raw_global_rules = [{**rule, "contexts": []} for rule in raw_global_rules]
            ge_cases = []

        if cfg["mode"] == "paper_flat":
            global_memory = "\n".join([*(f"Rule: {rule}" for rule in gs_rules), *(f"Case: {case}" for case in ge_cases)]) or "None."
            local_memory = "\n".join([*(f"Rule: {rule}" for rule in local_rules), *local_trajectories]) or "None."
        elif cfg["mode"] == "paper_no_coupling":
            global_memory = "\n".join(gs_rules + ge_cases) or "None."
            local_memory = "\n".join(local_rules + local_trajectories) or "None."
        else:
            global_memory = format_global_coupled_memory(raw_global_rules)
            local_memory = format_local_coupled_memory(local_rules, local_trajectories)

        return PREDICTION_AGENT_PROMPT.format(
            init_instruction=self.init_instruction,
            global_coupled_memory=global_memory,
            local_coupled_memory=local_memory,
            text=item["text"],
        )

    def run_all_experiments(self, test_data: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        memories = self._prepare_memories(test_data)
        all_metrics = {}
        all_token_stats = {}

        print(color_text("\n[Inference] Running ablation variants...", "cyan"))
        for variant_name, cfg in self.variants.items():
            print(color_text(f"\n>>> Running variant: {variant_name}", "yellow"))
            prompts = [self._build_prompt(item, memories[i], cfg) for i, item in enumerate(test_data)]
            outputs = generate_texts(self.llm, prompts, max_tokens=self.max_tokens, temperature=0.0)
            variant_tokens = get_token_stats(outputs)
            results = []
            for i, output in enumerate(outputs):
                pred = normalize_prediction(
                    parse_json_robust(output.outputs[0].text),
                    self.allowed_labels,
                    original_input_text=test_data[i]["text"],
                )
                results.append(
                    {
                        "input": test_data[i]["text"],
                        "gold": test_data[i].get("entities", []),
                        "pred": pred,
                        "raw_llm_output": output.outputs[0].text,
                    }
                )
            metrics = compute_micro_f1(results)
            all_metrics[variant_name] = metrics
            all_token_stats[variant_name] = {
                "final_inference_tokens_total": variant_tokens,
                "avg_final_inference_tokens_per_instance": round(variant_tokens / len(test_data), 2) if test_data else 0,
            }
            save_json(
                os.path.join(output_dir, f"res_{safe_name(variant_name)}.json"),
                {
                    "variant": variant_name,
                    "metrics": metrics,
                    "token": all_token_stats[variant_name],
                    "results": results,
                },
            )
            print(color_text(f"{variant_name}: F1={metrics['micro_f1']}", "green"))

        return {
            "metrics": all_metrics,
            "variant_token_stats": all_token_stats,
            "local_memory_tokens_total": self.local_token_usage,
        }
