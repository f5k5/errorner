from typing import Any, Dict, List


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if "text" not in item:
        item["text"] = item.get("sentence", item.get("input", ""))

    raw_entities = item.get("entities")
    if raw_entities is None:
        raw_entities = item.get("labels")
    if raw_entities is None:
        raw_entities = item.get("label")
    if raw_entities is None:
        raw_entities = item.get("gold")
    if raw_entities is None:
        raw_entities = []

    normalized: List[Dict[str, str]] = []
    if isinstance(raw_entities, dict):
        for entity_type, values in raw_entities.items():
            if isinstance(values, list):
                for text in values:
                    normalized.append({"entity_text": str(text), "entity_type": str(entity_type)})
    elif isinstance(raw_entities, list):
        for entity in raw_entities:
            if not isinstance(entity, dict):
                continue
            text = entity.get("entity_text") or entity.get("text") or entity.get("name") or entity.get("span")
            entity_type = entity.get("entity_type") or entity.get("type") or entity.get("label") or entity.get("entity")
            if text and entity_type:
                normalized.append({"entity_text": str(text), "entity_type": str(entity_type)})

    item["entities"] = normalized
    return item


def collect_labels(train_data: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            entity["entity_type"]
            for item in train_data
            for entity in item.get("entities", [])
            if entity.get("entity_type")
        }
    )
