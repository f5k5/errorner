from typing import Any, Dict, Iterable, List, Set, Tuple

from memoryner.utils.text import normalize_prediction


def entity_set(items: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    entities = normalize_prediction(items)
    return {(entity["entity_text"], entity["entity_type"]) for entity in entities}


def compute_micro_f1(results: List[Dict[str, Any]]) -> Dict[str, float]:
    tp, fp, fn = 0, 0, 0
    for item in results:
        gold = entity_set(item.get("gold", []))
        pred = entity_set(item.get("pred", []))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision * 100, 4),
        "recall": round(recall * 100, 4),
        "micro_f1": round(f1 * 100, 4),
    }
