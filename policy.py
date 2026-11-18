# policy.py
import json

FIELDS = {
    "role": {"type": "str", "values": ("admin", "editor", "viewer", "guest")},
    "user_dept": {"type": "str", "values": ("ENG", "FIN", "HR")},
    "resource_dept": {"type": "str", "values": ("ENG", "FIN", "HR")},
    "clearance": {"type": "int", "min": 0, "max": 3},
    "classification": {"type": "int", "min": 0, "max": 3},
    "resource_state": {"type": "str", "values": ("active", "archived", "locked")},
    "action": {"type": "str", "values": ("read", "edit", "delete")},
    "is_owner": {"type": "bool"},
    "is_delegate": {"type": "bool"},
    "mfa": {"type": "bool"},
    "emergency": {"type": "bool"},
    "network": {"type": "str", "values": ("corporate", "public")},
    "hour": {"type": "int", "min": 0, "max": 23},
    "purpose": {"type": "str", "values": ("routine", "incident")},
}
RULE_IDS = tuple(f"R{n:02d}" for n in range(1, 14))
SYSTEM_PROMPT = ('Evaluate the supplied synthetic access-control policy exactly. '
                 'Return one JSON object only, without Markdown fences or explanation. '
                 'Use exactly the keys decision and rule_id.')
POLICY_TEXT = "\n".join([
    "R01 DENY if resource_state == locked",
    "R02 DENY if clearance < classification",
    "R03 DENY if resource_state == archived and action in {edit, delete}",
    "R04 DENY if network == public and classification >= 2",
    "R05 DENY if action in {edit, delete} and not mfa",
    "R06 ALLOW if role == admin and emergency and purpose == incident and action in {read, edit}",
    "R07 DENY if action in {edit, delete} and not (9 <= hour <= 16)",
    "R08 ALLOW if role == admin",
    "R09 DENY if user_dept != resource_dept and not is_owner and not is_delegate",
    "R10 ALLOW if action == read and (role in {viewer, editor} or is_delegate)",
    "R11 ALLOW if action == edit and (role == editor or (role == viewer and (is_owner or is_delegate)))",
    "R12 ALLOW if action == delete and role == editor and is_owner",
    "R13 DENY by default",
])


def validate_request(r):
    if not isinstance(r, dict) or set(r) != set(FIELDS):
        raise ValueError("request must be a dict with exactly the 14 defined fields")
    for k, spec in FIELDS.items():
        v, t = r[k], spec["type"]
        if t == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{k} must be bool")
        elif t == "int":
            if isinstance(v, bool) or not isinstance(v, int) or not spec["min"] <= v <= spec["max"]:
                raise ValueError(f"{k} must be int in [{spec['min']},{spec['max']}]")
        else:
            if not isinstance(v, str) or v not in spec["values"]:
                raise ValueError(f"{k} must be one of {spec['values']}")
    return True


def _match(rid, r):
    return {
        "R01": r["resource_state"] == "locked",
        "R02": r["clearance"] < r["classification"],
        "R03": r["resource_state"] == "archived" and r["action"] in ("edit", "delete"),
        "R04": r["network"] == "public" and r["classification"] >= 2,
        "R05": r["action"] in ("edit", "delete") and not r["mfa"],
        "R06": r["role"] == "admin" and r["emergency"] and r["purpose"] == "incident" and r["action"] in ("read", "edit"),
        "R07": r["action"] in ("edit", "delete") and not (9 <= r["hour"] <= 16),
        "R08": r["role"] == "admin",
        "R09": r["user_dept"] != r["resource_dept"] and not r["is_owner"] and not r["is_delegate"],
        "R10": r["action"] == "read" and (r["role"] in ("viewer", "editor") or r["is_delegate"]),
        "R11": r["action"] == "edit" and (r["role"] == "editor" or (r["role"] == "viewer" and (r["is_owner"] or r["is_delegate"]))),
        "R12": r["action"] == "delete" and r["role"] == "editor" and r["is_owner"],
        "R13": True,
    }[rid]


def matching_rules(r):
    validate_request(r)
    return [rid for rid in RULE_IDS[:-1] if _match(rid, r)]


def evaluate(r):
    validate_request(r)
    allow = {"R06", "R08", "R10", "R11", "R12"}
    for rid in RULE_IDS:
        if _match(rid, r):
            return {"decision": "ALLOW" if rid in allow else "DENY", "rule_id": rid}
    return {"decision": "DENY", "rule_id": "R13"}


def canonical(r):
    validate_request(r)
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def make_prompt(r):
    return ("Access Control Rules (first matching rule wins):\n" + POLICY_TEXT +
            "\n\nRequest:\n" + canonical(r) +
            '\n\nReturn ONLY one JSON object with exactly two string fields: '
            'decision (ALLOW or DENY) and rule_id (R01 through R13). '
            'Do not include Markdown, code fences, reasoning, or any other text.')


def chat_messages(prompt):
    return [{'role':'system','content':SYSTEM_PROMPT}, {'role':'user','content':prompt}]
