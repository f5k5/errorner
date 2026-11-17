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

        index = 1
        for group_key, rules in rules_dict.items():
            for rule_object in rules:
                self.rule_id_map[index] = {
                    "group_key": rule_object.get("group_key", group_key),
                    "label": rule_object.get("label", group_key),
                    "category": rule_object.get("category", ""),
                    "rule": rule_object["rule"],
                    "contexts": rule_object.get("contexts", []),
                    "sources": rule_object.get("sources", []),
                }
                self.formatted_rules_list.append(
                    f"[{index}] (Error category: "
                    f"{self.rule_id_map[index]['category']}; "
                    f"target: {self.rule_id_map[index]['label']}): "
                    f"{rule_object['rule']}"
                )
                index += 1

        self.all_rules_text = "\n".join(self.formatted_rules_list)
        print(
            color_text(
                f"Loaded {len(self.formatted_rules_list)} global memories for retrieval.",
                "green",
            )
        )

    def retrieve(
        self,
        query_texts: List[str],
        top_k: int = 3,
    ) -> List[List[Dict[str, Any]]]:
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
        outputs = generate_texts(
            self.llm,
            prompts,
            max_tokens=64,
            temperature=0.0,
        )
        self.token_usage += get_token_stats(outputs)

        batch_results = []
        for output in outputs:
            ids = parse_json_robust(output.outputs[0].text)
            retrieved = []
            if isinstance(ids, list):
                valid_ids = [
                    int(value)
                    for value in ids
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.isdigit())
                ]
                for memory_id in valid_ids[:top_k]:
                    if memory_id in self.rule_id_map:
                        retrieved.append(self.rule_id_map[memory_id])
            batch_results.append(retrieved)
        return batch_results
