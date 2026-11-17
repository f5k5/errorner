import json
import random
import re
from typing import Any, Dict, List, Tuple

from memoryner.llm_engine import generate_texts, get_token_stats
from memoryner.memory.error_analysis import (
    classify_prediction_errors,
    introduces_new_errors,
    is_target_error_corrected,
)
from memoryner.prompts import (
    GLOBAL_REFLECTION_PROMPT,
    VERIFICATION_AGENT_PROMPT,
    ZERO_SHOT_PREDICTION_PROMPT,
)
from memoryner.utils.text import color_text, normalize_prediction, parse_json_robust


class GlobalRuleLearner:
    def __init__(
        self,
        llm: Any,
        labels_str: str,
        allowed_labels: List[str],
        domain_name: str,
    ):
        self.llm = llm
        self.labels_str = labels_str
        self.allowed_labels = allowed_labels
        self.domain_name = domain_name
        self.error_categories = [
            "Missed Entity",
            "Type Error",
            "Boundary Error",
            "Hallucination Error",
        ]
        # Keys are error-group identifiers. Type errors use a gold/predicted
        # type pair; the other categories use the relevant single entity type.
        self.global_rules_map_validated: Dict[str, List[Dict[str, Any]]] = {}
        self.global_rules_map_unvalidated: Dict[str, List[Dict[str, Any]]] = {}
        self.token_usage = 0
        self.max_retries = 3
        self.tau = 5
        self.lambda_global = 2
        self.init_instruction = f"Allowed entity types: {self.labels_str}"

    def _add_rule(
        self,
        target_map: Dict[str, List[Dict[str, Any]]],
        task: Dict[str, Any],
        rule: str,
        contexts: List[Dict[str, Any]],
    ) -> None:
        rule = rule.strip()
        if not rule or len(rule) < 5:
            return

        group_key = task["group_key"]
        entries = target_map.setdefault(group_key, [])
        existing = next(
            (item for item in entries if item.get("rule") == rule),
            None,
        )
        if existing is not None:
            known = {item.get("text") for item in existing.get("contexts", [])}
            for context in contexts:
                if context.get("text") not in known:
                    existing.setdefault("contexts", []).append(context)
                    existing.setdefault("sources", []).append(
                        context.get("text", "")
                    )
                    known.add(context.get("text"))
            return

        entries.append({
            "group_key": group_key,
            "label": task["label"],
            "category": task["category"],
            "rule": rule,
            "contexts": list(contexts),
            "sources": [context.get("text", "") for context in contexts],
        })

    def _classify_errors(
        self,
        samples: List[Dict[str, Any]],
        pred_outputs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        errors_by_group: Dict[str, List[Dict[str, Any]]] = {}

        for sample_index, output in enumerate(pred_outputs):
            sample = samples[sample_index]
            text = sample["text"]
            prediction = normalize_prediction(
                parse_json_robust(output.outputs[0].text),
                self.allowed_labels,
                original_input_text=text,
            )
            gold = normalize_prediction(
                sample.get("entities", []),
                self.allowed_labels,
            )
            errors = classify_prediction_errors(text, prediction, gold)

            for error_index, error in enumerate(errors):
                error["sample_index"] = sample_index
                error["error_index"] = error_index
                error["error_id"] = f"{sample_index}:{error_index}"
                error["full_gold"] = gold
                error["baseline_pred"] = prediction
                errors_by_group.setdefault(error["group_key"], []).append(error)

        return errors_by_group

    def _build_active_tasks(
        self,
        errors_by_group: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Keep only groups with at least tau distinct seed samples. Split by
        sample (not by individual error) so one sentence cannot occur in both
        the induction and independent verification subsets.

        The paper fixes tau=5 and lambda=2 but does not specify a split ratio;
        reserve lambda distinct samples for S_g,ver.
        """
        active_tasks = []
        for task_id, (group_key, pool) in enumerate(sorted(errors_by_group.items())):
            samples_by_id: Dict[Any, List[Dict[str, Any]]] = {}
            for error in pool:
                sample_id = error.get("text", error.get("sample_index"))
                samples_by_id.setdefault(sample_id, []).append(error)

            if len(samples_by_id) < self.tau:
                continue

            sample_ids = list(samples_by_id)
            random.shuffle(sample_ids)

            verification_size = self.lambda_global
            if len(sample_ids) <= verification_size:
                continue

            verification_ids = set(sample_ids[:verification_size])
            source_ids = set(sample_ids[verification_size:])

            source_pool = [
                error
                for sample_id, sample_errors in samples_by_id.items()
                if sample_id in source_ids
                for error in sample_errors
            ]
            verification_pool = [
                error
                for sample_id, sample_errors in samples_by_id.items()
                if sample_id in verification_ids
                for error in sample_errors
            ]

            if not source_pool or not verification_pool:
                continue

            first = pool[0]
            active_tasks.append({
                "task_id": task_id,
                "group_key": group_key,
                "label": first["group_label"],
                "category": first["category"],
                "source_pool": source_pool,
                "verification_pool": verification_pool,
                "source_sample_ids": source_ids,
                "verification_sample_ids": verification_ids,
            })

        return active_tasks

    @staticmethod
    def _context(error: Dict[str, Any], split: str) -> Dict[str, Any]:
        return {
            "text": error.get("text", ""),
            "gold": error.get("full_gold", error.get("gold", [])),
            "pred": error.get("baseline_pred", error.get("pred", [])),
            "error": error.get("msg", ""),
            "target_mention": error.get("target_mention", ""),
            "split": split,
        }

    @staticmethod
    def _extract_rules(raw_output: str) -> List[str]:
        parsed = parse_json_robust(raw_output)
        if isinstance(parsed, str):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []

        rules = []
        for rule_text in parsed[:2]:
            if not isinstance(rule_text, str):
                continue
            rule_text = re.sub(
                r"^(?:Rule\s*\d+:?|[-*•]|\d+\.)\s*",
                "",
                rule_text.strip(),
                flags=re.IGNORECASE,
            ).strip(chr(34) + "'")
            if len(rule_text) >= 5 and rule_text not in rules:
                rules.append(rule_text)
        return rules

    def _evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        validation_prompts = []
        validation_meta = []

        for candidate_index, candidate in enumerate(candidates):
            task = candidate["task"]
            for split_name, pool in (
                ("source", task["source_pool"]),
                ("verification", task["verification_pool"]),
            ):
                for error in pool:
                    validation_prompts.append(
                        VERIFICATION_AGENT_PROMPT.format(
                            entity_type_definitions=self.labels_str,
                            r=candidate["rule"],
                            X_i=error["text"],
                        )
                    )
                    validation_meta.append({
                        "candidate_index": candidate_index,
                        "split": split_name,
                        "error": error,
                    })

        outputs = generate_texts(
            self.llm,
            validation_prompts,
            max_tokens=1024,
            temperature=0.0,
        )
        self.token_usage += get_token_stats(outputs)

        evaluations = {
            index: {
                "source_corrected": [],
                "verification_unseen_corrected": [],
            }
            for index in range(len(candidates))
        }

        for meta, output in zip(validation_meta, outputs):
            candidate_index = meta["candidate_index"]
            error = meta["error"]
            new_prediction = normalize_prediction(
                parse_json_robust(output.outputs[0].text),
                self.allowed_labels,
                original_input_text=error["text"],
            )

            corrected = (
                is_target_error_corrected(error, new_prediction)
                and not introduces_new_errors(
                    error["baseline_pred"],
                    new_prediction,
                    error["full_gold"],
                )
            )
            if not corrected:
                continue

            if meta["split"] == "source":
                evaluations[candidate_index]["source_corrected"].append(error)
            else:
                source_mentions = {
                    item.get("target_mention", "")
                    for item in candidates[candidate_index]["task"]["source_pool"]
                }
                target_mention = error.get("target_mention", "")
                if target_mention and target_mention not in source_mentions:
                    evaluations[candidate_index][
                        "verification_unseen_corrected"
                    ].append(error)

        return evaluations

    def learn_from_data(
        self,
        practice_data: List[Dict[str, Any]],
        sample_size: int = 200,
    ) -> Tuple[Dict[str, Any], int]:
        print(
            color_text(
                f"\n[Global] Error analysis on {sample_size} samples...",
                "yellow",
            )
        )
        samples = (
            random.sample(practice_data, sample_size)
            if len(practice_data) > sample_size
            else list(practice_data)
        )

        prediction_prompts = [
            ZERO_SHOT_PREDICTION_PROMPT.format(
                init_instruction=self.init_instruction,
                text=item["text"],
            )
            for item in samples
        ]
        prediction_outputs = generate_texts(
            self.llm,
            prediction_prompts,
            max_tokens=1024,
            temperature=0.0,
        )
        self.token_usage += get_token_stats(prediction_outputs)

        errors_by_group = self._classify_errors(samples, prediction_outputs)
        active_tasks = self._build_active_tasks(errors_by_group)

        print(
            color_text(
                f"[Global] Rule synthesis and validation: "
                f"{len(active_tasks)} eligible error groups "
                f"(tau={self.tau}, lambda={self.lambda_global})",
                "yellow",
            )
        )

        for attempt in range(self.max_retries):
            if not active_tasks:
                break

            print(
                color_text(
                    f"Global attempt {attempt + 1}/{self.max_retries}, "
                    f"active groups={len(active_tasks)}",
                    "cyan",
                )
            )

            synthesis_prompts = []
            synthesis_refs = []
            for task in active_tasks:
                source_examples = task["source_pool"]
                global_episodic_memory = "\n".join(
                    (
                        f"- Error: {error['msg']}\n"
                        f"  Text: {error['text']}\n"
                        f"  Gold: {json.dumps(error['full_gold'], ensure_ascii=False)}\n"
                        f"  Zero-shot prediction: "
                        f"{json.dumps(error['baseline_pred'], ensure_ascii=False)}"
                    )
                    for error in source_examples
                )
                synthesis_prompts.append(
                    GLOBAL_REFLECTION_PROMPT.format(
                        entity_type=task["label"],
                        error_type=task["category"],
                        global_episodic_memory=global_episodic_memory,
                    )
                )
                synthesis_refs.append(task)

            rule_outputs = generate_texts(
                self.llm,
                synthesis_prompts,
                max_tokens=256,
                temperature=0.3,
            )
            self.token_usage += get_token_stats(rule_outputs)

            candidates = []
            for task, output in zip(synthesis_refs, rule_outputs):
                for rule_text in self._extract_rules(output.outputs[0].text):
                    source_contexts = [
                        self._context(error, "source")
                        for error in task["source_pool"]
                    ]
                    self._add_rule(
                        self.global_rules_map_unvalidated,
                        task,
                        rule_text,
                        source_contexts,
                    )
                    candidates.append({
                        "task": task,
                        "rule": rule_text,
                    })

            if not candidates:
                continue

            evaluations = self._evaluate_candidates(candidates)
            completed_task_ids = set()

            for candidate_index, candidate in enumerate(candidates):
                task = candidate["task"]
                evaluation = evaluations[candidate_index]
                source_corrected = evaluation["source_corrected"]
                verification_unseen = evaluation["verification_unseen_corrected"]

                passes_source = (
                    len(source_corrected) > len(task["source_pool"]) / 2
                )
                passes_verification = (
                    len(verification_unseen) >= self.lambda_global
                )
                if not (passes_source and passes_verification):
                    continue

                support_contexts = [
                    self._context(error, "source")
                    for error in source_corrected
                ] + [
                    self._context(error, "verification")
                    for error in verification_unseen
                ]
                self._add_rule(
                    self.global_rules_map_validated,
                    task,
                    candidate["rule"],
                    support_contexts,
                )
                completed_task_ids.add(task["task_id"])

            # Keep the existing bounded retry behavior for groups for which
            # none of the generated candidates passed the paper's criteria.
            active_tasks = [
                task for task in active_tasks
                if task["task_id"] not in completed_task_ids
            ]

        validated_count = sum(
            len(items) for items in self.global_rules_map_validated.values()
        )
        candidate_count = sum(
            len(items) for items in self.global_rules_map_unvalidated.values()
        )
        print(
            color_text(
                f"[Global] Candidate rules={candidate_count}, "
                f"validated rules={validated_count}",
                "green",
            )
        )

        return {
            "validated": self.global_rules_map_validated,
            "unvalidated": self.global_rules_map_unvalidated,
        }, self.token_usage
