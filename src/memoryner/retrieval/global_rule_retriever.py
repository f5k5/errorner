from typing import Any, Dict, List

from memoryner.llm_engine import generate_texts, get_token_stats
from memoryner.prompts import LLM_RETRIEVAL_PROMPT
from memoryner.utils.text import color_text, parse_json_robust


class LLMRuleRetriever:
    def __init__(self, rules_dict: Dict[str, List[Dict[str, Any]]], llm: Any):
        self.llm = llm
        self.rule_id_map: Dict[int, Dict[str, Any]] = {}
        self.formatted_rules_list: List[str] = []
        self.token_usage = 0

        idx = 1
        for label, rules in rules_dict.items():
            for rule_object in rules:
                self.rule_id_map[idx] = {
                    "label": label,
                    "rule": rule_object["rule"],
                    "contexts": rule_object.get("contexts", []),
                    "sources": rule_object.get("sources", []),
                }
                self.formatted_rules_list.append(f"[{idx}] (Targeting {label}): {rule_object['rule']}")
                idx += 1

        self.all_rules_text = "\n".join(self.formatted_rules_list)
        print(color_text(f"Loaded {len(self.formatted_rules_list)} global memories for retrieval.", "green"))

    def retrieve(self, query_texts: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        if not self.rule_id_map:
            return [[] for _ in query_texts]

        prompts = [
            LLM_RETRIEVAL_PROMPT.format(
                all_memories_numbered=self.all_rules_text,
                k_global=top_k,
                text=text,
            )
            for text in query_texts
        ]
        outputs = generate_texts(self.llm, prompts, max_tokens=64, temperature=0.0)
        self.token_usage += get_token_stats(outputs)

        batch_results: List[List[Dict[str, Any]]] = []
        for output in outputs:
            ids = parse_json_robust(output.outputs[0].text)
            retrieved: List[Dict[str, Any]] = []
            if isinstance(ids, list):
                valid_ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
                for memory_id in valid_ids[:top_k]:
                    if memory_id in self.rule_id_map:
                        retrieved.append(self.rule_id_map[memory_id])
            batch_results.append(retrieved)
        return batch_results
