import json
from typing import Any, Dict, List


def format_global_coupled_memory(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "None."
    chunks: List[str] = []
    for idx, rule in enumerate(rules, start=1):
        rule_text = rule.get("rule", "")
        label = rule.get("label", "")
        contexts = rule.get("contexts", [])
        episodic_lines = []
        for case_id, context in enumerate(contexts[:3], start=1):
            episodic_lines.append(
                f"Episodic Case {case_id}:\n"
                f"Text: {context.get('text', '')}\n"
                f"Gold Output: {json.dumps(context.get('gold', []), ensure_ascii=False)}"
            )
        episodic_block = "\n".join(episodic_lines) if episodic_lines else "No episodic case."
        chunks.append(
            f"[Global Memory {idx}]\n"
            f"Target Type: {label}\n"
            f"Semantic Rule: {rule_text}\n"
            f"Supporting Episodic Memory:\n{episodic_block}"
        )
    return "\n\n".join(chunks)


def format_local_coupled_memory(local_rules: List[str], trajectories: List[str]) -> str:
    if not local_rules and not trajectories:
        return "None."
    chunks: List[str] = []
    if local_rules:
        chunks.append("Local Semantic Memory:\n" + "\n".join(f"- {rule}" for rule in local_rules))
    if trajectories:
        chunks.append("Local Episodic Memory:\n" + "\n\n".join(trajectories))
    return "\n\n".join(chunks)
