import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence


def color_text(text: str, color: str) -> str:
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "end": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())


def parse_json_robust(text: Optional[str]) -> Any:
    if text is None:
        return []
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start_idx = stripped.find("[")
    end_idx = stripped.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(stripped[start_idx : end_idx + 1])
        except Exception:
            pass
    start_idx = stripped.find("{")
    end_idx = stripped.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(stripped[start_idx : end_idx + 1])
        except Exception:
            pass
    return []


def align_text_case(original_text: str, extracted_text: str) -> str:
    if not original_text or not extracted_text:
        return extracted_text
    try:
        match = re.search(re.escape(extracted_text), original_text, re.IGNORECASE)
        return match.group(0) if match else extracted_text
    except Exception:
        return extracted_text


def normalize_prediction(
    pred_data: Any,
    allowed_labels: Optional[Sequence[str]] = None,
    original_input_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    if isinstance(pred_data, dict):
        if isinstance(pred_data.get("entities"), list):
            pred_data = pred_data["entities"]
        else:
            pred_data = [pred_data]
    if not isinstance(pred_data, list):
        return []

    allowed_set = {x.lower().strip() for x in allowed_labels} if allowed_labels else None
    valid_entities: List[Dict[str, str]] = []
    seen = set()

    for item in pred_data:
        if not isinstance(item, dict):
            continue
        txt = item.get("entity_text") or item.get("text") or item.get("name") or item.get("span")
        typ = item.get("entity_type") or item.get("type") or item.get("label") or item.get("entity")
        if not txt or not typ:
            continue
        txt = str(txt).strip()
        typ_clean = str(typ).strip()
        if original_input_text:
            txt = align_text_case(original_input_text, txt)
        typ_lower = typ_clean.lower()
        if allowed_set and typ_lower not in allowed_set:
            continue
        final_type = typ_clean
        if allowed_labels:
            for label in allowed_labels:
                if label.lower().strip() == typ_lower:
                    final_type = label
                    break
        key = (txt, final_type)
        if key in seen:
            continue
        seen.add(key)
        valid_entities.append({"entity_text": txt, "entity_type": final_type})
    return valid_entities
