import json
from typing import Any, Dict, List


def format_global_coupled_memory(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "None."
    chunks = []
    for index, rule in enumerate(rules, start=1):
        rule_text = rule.get("rule", "")
        label = rule.get("label", "")
        category = rule.get("category", "")
        contexts = rule.get("contexts", [])
        episodic_lines = []
        for case_id, context in enumerate(contexts[:3], start=1):
            episodic_lines.append(
                f"Episodic Case {case_id}:\n"
                f"Text: {context.get('text', '')}\n"
                f"Gold Output: "
                f"{json.dumps(context.get('gold', []), ensure_ascii=False)}\n"
                f"Zero-shot Prediction: "
                f"{json.dumps(context.get('pred', []), ensure_ascii=False)}\n"
                f"Prediction Feedback: {context.get('error', '')}"
            )
        episodic_block = (
            "\n".join(episodic_lines)
            if episodic_lines else "No episodic case."
        )
        chunks.append(
            f"[Global Memory {index}]\n"
            f"Error Category: {category}\n"
            f"Target Type: {label}\n"
            f"Semantic Rule: {rule_text}\n"
            f"Supporting Episodic Memory:\n{episodic_block}"
        )
    return "\n\n".join(chunks)


def format_local_coupled_memory(
    local_rules: List[Dict[str, Any]],
    trajectories: List[str],
) -> str:
    """
    Present each local semantic rule together with the episodic sample
    from which that rule was induced, preserving the mapping.
    """
    if not local_rules and not trajectories:
        return "None."

    chunks = []
    if local_rules:
        rule_blocks = []
        for index, item in enumerate(local_rules, start=1):
            if isinstance(item, dict):
                rule_text = item.get("rule", "")
                support_case = item.get("support_case", {})
            else:
                # Backward compatibility for older serialized memory objects.
                rule_text = str(item)
                support_case = {}

            support_text = support_case.get("text", "")
            support_gold = support_case.get("gold", [])
            support_pred = support_case.get("pred", [])
            support_feedback = support_case.get(
                "critique",
                support_case.get("error", ""),
            )

            if support_text:
                support = (
                    f"Supporting Episodic Case:\n"
                    f"Text: {support_text}\n"
                    f"Zero-shot Prediction: "
                    f"{json.dumps(support_pred, ensure_ascii=False)}\n"
                    f"Gold Output: "
                    f"{json.dumps(support_gold, ensure_ascii=False)}\n"
                    f"Prediction Feedback: {support_feedback}"
                )
            else:
                support = "No paired episodic case."

            rule_blocks.append(
                f"[Local Memory {index}]\n"
                f"Semantic Rule: {rule_text}\n"
                f"{support}"
            )
        chunks.append("\n\n".join(rule_blocks))

    if trajectories:
        chunks.append(
            "Local Episodic Memory:\n" + "\n\n".join(trajectories)
        )

    return "\n\n".join(chunks)
