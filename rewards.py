# rewards.py
import json
from policy import RULE_IDS, evaluate

_ALLOWED = {"decision": ("ALLOW", "DENY"), "rule_id": RULE_IDS}


def _no_dup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate key")
        d[k] = v
    return d


def parse_output(text):
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text, object_pairs_hook=_no_dup)
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != {"decision", "rule_id"}:
        return None
    for k in ("decision", "rule_id"):
        if not isinstance(obj[k], str) or obj[k] not in _ALLOWED[k]:
            return None
    return {"decision": obj["decision"], "rule_id": obj["rule_id"]}


def completion_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list) and len(c) == 1 and isinstance(c[0], dict) and c[0].get("role") == "assistant":
        content = c[0].get("content", "")
        return content if isinstance(content, str) else None
    return None


def score(text, request):
    got = parse_output(text)
    if got is None:
        return 0.0
    exp = evaluate(request)
    if got == exp:
        return 1.0
    if got["decision"] == exp["decision"]:
        return 0.4
    return 0.0


def access_reward(completions, request_json, **kwargs):
    """TRL: request_json is a parallel list of JSON strings, one per completion."""
    if isinstance(request_json, str):
        request_json = [request_json]
    if len(completions) != len(request_json):
        raise ValueError(f"length mismatch: {len(completions)} completions vs {len(request_json)} requests")
    rewards = []
    for comp, req_str in zip(completions, request_json):
        req = json.loads(req_str) if isinstance(req_str, str) else req_str
        t = completion_text(comp)
        rewards.append(score(t, req) if t is not None else 0.0)
    return rewards
