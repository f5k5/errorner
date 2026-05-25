import json
import random
import re
from typing import Any, Dict, List, Tuple

from memoryner.llm_engine import generate_texts, get_token_stats
from memoryner.prompts import GLOBAL_REFLECTION_PROMPT, VERIFICATION_AGENT_PROMPT, ZERO_SHOT_PREDICTION_PROMPT
from memoryner.utils.text import color_text, normalize_prediction, parse_json_robust


class GlobalRuleLearner:
    def __init__(self, llm: Any, labels_str: str, allowed_labels: List[str], domain_name: str):
        self.llm = llm
        self.labels_str = labels_str
        self.allowed_labels = allowed_labels
        self.domain_name = domain_name
        self.error_categories = ["Missed Entity", "Type Error", "Boundary Error", "Hallucination Error"]
        self.global_rules_map_validated = {label: [] for label in allowed_labels}
        self.global_rules_map_unvalidated = {label: [] for label in allowed_labels}
        self.token_usage = 0
        self.max_retries = 3
        self.init_instruction = f"Allowed entity types: {self.labels_str}"

    def _add_rule(self, target_map, label: str, category: str, rule: str, contexts: List[Dict[str, Any]], sources: List[str]) -> None:
        rule = rule.strip()
        if not rule or len(rule) < 5:
            return
        existing = {item["rule"] for item in target_map[label]}
        if rule in existing:
            return
        target_map[label].append(
            {
                "label": label,
                "category": category,
                "rule": rule,
                "contexts": contexts,
                "sources": sources,
            }
        )

    def _classify_errors(self, samples: List[Dict[str, Any]], pred_outputs) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        errors_by_label_cat = {
            label: {category: [] for category in self.error_categories}
            for label in self.allowed_labels
        }

        for i, output in enumerate(pred_outputs):
            text = samples[i]["text"]
            pred = normalize_prediction(
                parse_json_robust(output.outputs[0].text),
                self.allowed_labels,
                original_input_text=text,
            )
            gold = normalize_prediction(samples[i].get("entities", []), self.allowed_labels)
            unmatched_preds = list(pred)

            for gold_entity in gold:
                gold_text = gold_entity["entity_text"]
                gold_type = gold_entity["entity_type"]
                exact_match = next((p for p in unmatched_preds if p["entity_text"] == gold_text), None)
                if exact_match:
                    unmatched_preds.remove(exact_match)
                    pred_type = exact_match["entity_type"]
                    if pred_type != gold_type:
                        err_data = {
                            "msg": f"Predicted type {pred_type}, correct type {gold_type}, entity {gold_text}",
                            "text": text,
                            "gold": gold_entity,
                            "pred": exact_match,
                            "full_gold": gold,
                        }
                        if gold_type in errors_by_label_cat:
                            errors_by_label_cat[gold_type]["Type Error"].append(err_data)
                        if pred_type in errors_by_label_cat:
                            errors_by_label_cat[pred_type]["Type Error"].append(err_data)
                    continue

                boundary_match = next(
                    (
                        p
                        for p in unmatched_preds
                        if gold_text in p["entity_text"] or p["entity_text"] in gold_text
                    ),
                    None,
                )
                if boundary_match:
                    unmatched_preds.remove(boundary_match)
                    err_data = {
                        "msg": (
                            f"Predicted boundary {boundary_match['entity_text']} ({boundary_match['entity_type']}), "
                            f"correct boundary {gold_text} ({gold_type})"
                        ),
                        "text": text,
                        "gold": gold_entity,
                        "pred": boundary_match,
                        "full_gold": gold,
                    }
                    if gold_type in errors_by_label_cat:
                        errors_by_label_cat[gold_type]["Boundary Error"].append(err_data)
                    if boundary_match["entity_type"] in errors_by_label_cat:
                        errors_by_label_cat[boundary_match["entity_type"]]["Boundary Error"].append(err_data)
                    continue

                err_data = {
                    "msg": f"Entity {gold_text} ({gold_type}) was missed.",
                    "text": text,
                    "gold": gold_entity,
                    "pred": None,
                    "full_gold": gold,
                }
                if gold_type in errors_by_label_cat:
                    errors_by_label_cat[gold_type]["Missed Entity"].append(err_data)

            for pred_entity in unmatched_preds:
                pred_type = pred_entity["entity_type"]
                err_data = {
                    "msg": f"Predicted {pred_entity['entity_text']} ({pred_type}) but it is not an entity.",
                    "text": text,
                    "gold": None,
                    "pred": pred_entity,
                    "full_gold": gold,
                }
                if pred_type in errors_by_label_cat:
                    errors_by_label_cat[pred_type]["Hallucination Error"].append(err_data)

        return errors_by_label_cat

    def _build_active_tasks(self, errors_by_label_cat: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        active_tasks = []
        task_id = 0
        for label in self.allowed_labels:
            for category in self.error_categories:
                pool = errors_by_label_cat[label][category]
                if not pool:
                    continue
                active_tasks.append(
                    {
                        "task_id": task_id,
                        "label": label,
                        "category": category,
                        "original_pool": pool,
                        "unresolved_pool": list(pool),
                    }
                )
                task_id += 1
        return active_tasks

    def learn_from_data(self, practice_data: List[Dict[str, Any]], sample_size: int = 200) -> Tuple[Dict[str, Any], int]:
        print(color_text(f"\n[Global] Error analysis on {sample_size} samples...", "yellow"))
        samples = random.sample(practice_data, sample_size) if len(practice_data) > sample_size else practice_data

        pred_prompts = [
            ZERO_SHOT_PREDICTION_PROMPT.format(init_instruction=self.init_instruction, text=item["text"])
            for item in samples
        ]
        pred_outputs = generate_texts(self.llm, pred_prompts, max_tokens=1024, temperature=0.0)
        self.token_usage += get_token_stats(pred_outputs)

        errors_by_label_cat = self._classify_errors(samples, pred_outputs)
        active_tasks = self._build_active_tasks(errors_by_label_cat)

        print(color_text("[Global] Rule synthesis and validation...", "yellow"))
        for attempt in range(self.max_retries):
            if not active_tasks:
                break
            print(color_text(f"Global attempt {attempt + 1}/{self.max_retries}, active groups={len(active_tasks)}", "cyan"))

            synthesis_prompts = []
            synthesis_refs = []
            for task in active_tasks:
                subset = random.sample(task["unresolved_pool"], min(10, len(task["unresolved_pool"])))
                global_episodic_memory = "\n".join(
                    f"- Error: {err['msg']}\n  Text: {err['text']}\n  Gold: {json.dumps(err['full_gold'], ensure_ascii=False)}"
                    for err in subset
                )
                synthesis_prompts.append(
                    GLOBAL_REFLECTION_PROMPT.format(
                        entity_type=task["label"],
                        error_type=task["category"],
                        global_episodic_memory=global_episodic_memory,
                    )
                )
                synthesis_refs.append((task, subset))

            rule_outputs = generate_texts(self.llm, synthesis_prompts, max_tokens=256, temperature=0.3)
            self.token_usage += get_token_stats(rule_outputs)

            candidates = []
            for i, output in enumerate(rule_outputs):
                task, subset = synthesis_refs[i]
                parsed = parse_json_robust(output.outputs[0].text)
                if isinstance(parsed, str):
                    parsed = [parsed]
                if not isinstance(parsed, list):
                    parsed = []
                for rule_text in parsed[:2]:
                    if not isinstance(rule_text, str):
                        continue
                    rule_text = re.sub(r"^(?:Rule\s*\d+:?|[-*•]|\d+\.)\s*", "", rule_text.strip(), flags=re.IGNORECASE).strip("\"'")
                    if len(rule_text) < 5:
                        continue
                    contexts = [{"text": err["text"], "gold": err["full_gold"]} for err in subset[:3]]
                    sources = [err["text"] for err in subset[:3]]
                    self._add_rule(self.global_rules_map_unvalidated, task["label"], task["category"], rule_text, contexts, sources)
                    candidates.append({"task": task, "rule": rule_text})

            if not candidates:
                continue

            validation_prompts = []
            validation_meta = []
            for candidate_id, candidate in enumerate(candidates):
                task = candidate["task"]
                for err_data in task["original_pool"]:
                    validation_prompts.append(
                        VERIFICATION_AGENT_PROMPT.format(
                            entity_type_definitions=self.labels_str,
                            r=candidate["rule"],
                            X_i=err_data["text"],
                        )
                    )
                    validation_meta.append({"candidate_id": candidate_id, "err_data": err_data})

            validation_outputs = generate_texts(self.llm, validation_prompts, max_tokens=1024, temperature=0.0)
            self.token_usage += get_token_stats(validation_outputs)

            fixed_by_candidate = {i: [] for i in range(len(candidates))}
            for output_id, validation_output in enumerate(validation_outputs):
                meta = validation_meta[output_id]
                err_data = meta["err_data"]
                new_pred = normalize_prediction(
                    parse_json_robust(validation_output.outputs[0].text),
                    self.allowed_labels,
                    original_input_text=err_data["text"],
                )
                gold_entity = err_data.get("gold")
                is_fixed = False
                if gold_entity:
                    is_fixed = any(
                        pred["entity_text"] == gold_entity["entity_text"]
                        and pred["entity_type"] == gold_entity["entity_type"]
                        for pred in new_pred
                    )
                else:
                    bad_pred = err_data.get("pred")
                    if bad_pred:
                        is_fixed = not any(pred["entity_text"] == bad_pred["entity_text"] for pred in new_pred)
                if is_fixed:
                    fixed_by_candidate[meta["candidate_id"]].append(err_data)

            fixed_texts_by_task: Dict[int, set] = {}
            for candidate_id, fixed_errors in fixed_by_candidate.items():
                if not fixed_errors:
                    continue
                candidate = candidates[candidate_id]
                task = candidate["task"]
                contexts = [{"text": err["text"], "gold": err["full_gold"]} for err in fixed_errors]
                sources = [err["text"] for err in fixed_errors]
                self._add_rule(
                    self.global_rules_map_validated,
                    task["label"],
                    task["category"],
                    candidate["rule"],
                    contexts,
                    sources,
                )
                fixed_texts_by_task.setdefault(task["task_id"], set()).update(sources)

            next_active_tasks = []
            for task in active_tasks:
                fixed_texts = fixed_texts_by_task.get(task["task_id"], set())
                if fixed_texts:
                    task["unresolved_pool"] = [err for err in task["unresolved_pool"] if err["text"] not in fixed_texts]
                if task["unresolved_pool"]:
                    next_active_tasks.append(task)
            active_tasks = next_active_tasks

        return {
            "validated": self.global_rules_map_validated,
            "unvalidated": self.global_rules_map_unvalidated,
        }, self.token_usage
