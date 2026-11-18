# dataset.py
import csv, hashlib, json, random, sys
from pathlib import Path
from policy import FIELDS, RULE_IDS, validate_request, canonical, matching_rules, evaluate, make_prompt

QUOTAS = {"train": 80, "validation": 20, "test": 20}
_PER_RULE = sum(QUOTAS.values())
HOUR_BOUNDARIES = (8, 9, 16, 17)
COLUMNS = (["example_id","split","difficulty","prompt","request_json","expected_output",
            "expected_decision","expected_rule_id","matching_rule_ids","conflicting_rules"]
           + list(FIELDS.keys()))


def _random_fields(rng):
    return {
        "role": rng.choice(FIELDS["role"]["values"]),
        "user_dept": rng.choice(FIELDS["user_dept"]["values"]),
        "resource_dept": rng.choice(FIELDS["resource_dept"]["values"]),
        "clearance": rng.randint(0,3), "classification": rng.randint(0,3),
        "resource_state": rng.choice(FIELDS["resource_state"]["values"]),
        "action": rng.choice(FIELDS["action"]["values"]),
        "is_owner": rng.random()<0.5, "is_delegate": rng.random()<0.5,
        "mfa": rng.random()<0.5, "emergency": rng.random()<0.3,
        "network": rng.choice(FIELDS["network"]["values"]),
        "hour": rng.randint(0,23), "purpose": rng.choice(FIELDS["purpose"]["values"]),
    }


def _difficulty(r):
    if r["hour"] in HOUR_BOUNDARIES: return "advanced"
    if r["emergency"] and r["purpose"]=="incident": return "advanced"
    m = matching_rules(r); A={"R06","R08","R10","R11","R12"}
    if any(x in A for x in m) and any(x not in A for x in m): return "advanced"
    return "standard"


def _conflicting(r):
    m = matching_rules(r); A={"R06","R08","R10","R11","R12"}
    return "|".join(m) if (any(x in A for x in m) and any(x not in A for x in m)) else ""


def _make_row(r, split, idx):
    ev = evaluate(r); mids = matching_rules(r)
    return {"example_id": f"{split}_{idx:04d}", "split": split, "difficulty": _difficulty(r),
            "prompt": make_prompt(r), "request_json": canonical(r),
            "expected_output": json.dumps({"decision":ev["decision"],"rule_id":ev["rule_id"]},sort_keys=True,separators=(",",":")),
            "expected_decision": ev["decision"], "expected_rule_id": ev["rule_id"],
            "matching_rule_ids": "|".join(mids), "conflicting_rules": _conflicting(r), **r}


def generate(seed=173):
    rng = random.Random(seed); pools={r:[] for r in RULE_IDS}; seen=set(); attempts=0
    while any(len(pools[r])<_PER_RULE for r in RULE_IDS):
        attempts+=1
        if attempts>2_000_000:
            missing={r:_PER_RULE-len(pools[r]) for r in RULE_IDS if len(pools[r])<_PER_RULE}
            raise RuntimeError(f"Quota not met after {attempts} attempts: {missing}")
        r=_random_fields(rng); key=canonical(r)
        if key in seen: continue
        wr=evaluate(r)["rule_id"]
        if len(pools[wr])>=_PER_RULE: continue
        seen.add(key); pools[wr].append(r)
    splits={"train":[],"validation":[],"test":[]}
    for rid in RULE_IDS:
        ex=pools[rid]; rng.shuffle(ex)
        for i,r in enumerate(ex):
            sp="train" if i<QUOTAS["train"] else ("validation" if i<QUOTAS["train"]+QUOTAS["validation"] else "test")
            splits[sp].append(_make_row(r, sp, i))
    for sp in splits: rng.shuffle(splits[sp])
    # re-assign sequential ids after shuffle
    for sp,rows in splits.items():
        for i,row in enumerate(rows): row["example_id"]=f"{sp}_{i:04d}"
    return splits, attempts


def _coerce(row):
    r={}
    for k,spec in FIELDS.items():
        v=row[k]
        if spec["type"]=="bool" and isinstance(v,str):
            if v not in ("True", "true", "False", "false"):
                raise ValueError(f"Invalid boolean feature: {k}={v!r}")
            r[k]=v in("True","true")
        elif spec["type"]=="int" and isinstance(v,str): r[k]=int(v)
        else: r[k]=v
    return r


def validate(splits):
    if set(splits) != set(QUOTAS):
        raise ValueError('Expected train, validation, and test splits')
    issues=[]; seen=set(); ids=set(); per={r:{s:0 for s in QUOTAS} for r in RULE_IDS}
    for sp,rows in splits.items():
        for row in rows:
            r=_coerce(row)
            validate_request(r)
            ev=evaluate(r)
            if row['split'] != sp: issues.append('split mismatch')
            if json.loads(row['expected_output']) != ev: issues.append('expected_output mismatch')
            if ev["rule_id"]!=row["expected_rule_id"]: issues.append(f"{row['example_id']}: rule mismatch")
            if ev["decision"]!=row["expected_decision"]: issues.append(f"{row['example_id']}: decision mismatch")
            if make_prompt(r)!=row["prompt"]: issues.append(f"{row['example_id']}: prompt mismatch")
            k=canonical(r)
            if canonical(json.loads(row['request_json'])) != k: issues.append('request_json/feature mismatch')
            if row['difficulty'] != _difficulty(r): issues.append('difficulty mismatch')
            if row['matching_rule_ids'] != '|'.join(matching_rules(r)): issues.append('matching rules mismatch')
            if row['conflicting_rules'] != _conflicting(r): issues.append('conflict metadata mismatch')
            if k in seen: issues.append(f"{row['example_id']}: duplicate")
            if row['example_id'] in ids: issues.append('duplicate example_id')
            ids.add(row['example_id'])
            seen.add(k); per[row["expected_rule_id"]][sp]+=1
    for rid in RULE_IDS:
        for sp,q in QUOTAS.items():
            if per[rid][sp]!=q: issues.append(f"quota {rid}/{sp}: {per[rid][sp]}!={q}")
    if issues:
        raise ValueError('; '.join(issues[:20]))
    return {"issues":issues,"total":sum(len(v) for v in splits.values()),"ok":not issues,
            "counts":{sp:len(rows) for sp,rows in splits.items()},"per_rule":per}


def load_rows(path):
    with open(path,newline="",encoding="utf-8") as f: return list(csv.DictReader(f))


def write_csv(path, rows, columns=None):
    cols=columns or COLUMNS
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        for row in rows: w.writerow(row)


def audit_rows(train):
    sel=[]; ids=set()
    def take(predicate, label):
        row=next((r for r in train if r['example_id'] not in ids and predicate(r)),None)
        if row is None: raise RuntimeError(f'Audit: no unused example for {label}')
        ids.add(row['example_id']); sel.append(dict(row,human_verified='',review_notes=''))
    for rid in RULE_IDS:
        take(lambda r:r['expected_rule_id']==rid,rid)
    for h in HOUR_BOUNDARIES:
        take(lambda r:_coerce(r)['hour']==h and r['action'] in ('edit','delete'),f'hour {h}')
    take(lambda r:r['expected_rule_id']=='R06' and not 9<=_coerce(r)['hour']<=16,'emergency override')
    take(lambda r:r['expected_rule_id'] in ('R01','R02','R03','R04','R05') and _coerce(r)['emergency'] and r['purpose']=='incident','blocked emergency')
    take(lambda r:bool(r['conflicting_rules']),'policy conflict')
    return sel


def main():
    import argparse
    ap=argparse.ArgumentParser(prog="dataset.py")
    ap.add_argument("--output",default="data")
    ap.add_argument("--seed",type=int,default=173)
    ap.add_argument("--validate-only",action="store_true")
    args=ap.parse_args()
    out=Path(args.output)
    if args.validate_only:
        splits={}
        for sp in("train","validation","test"):
            p=out/f"{sp}.csv"
            if not p.exists(): print(f"ERROR: {p} missing",file=sys.stderr); sys.exit(1)
            splits[sp]=load_rows(p)
        s=validate(splits); print(json.dumps(s,indent=2)); sys.exit(0 if s["ok"] else 1)
    out.mkdir(parents=True,exist_ok=True)
    splits,attempts=generate(seed=args.seed)
    validate(splits)
    print(f"Generated {sum(len(v) for v in splits.values())} examples in {attempts} attempts")
    for sp in("train","validation","test"): write_csv(out/f"{sp}.csv",splits[sp])
    audit=audit_rows(splits["train"])
    acols=COLUMNS+["human_verified","review_notes"]
    for row in audit: row["human_verified"]=""; row["review_notes"]=""
    write_csv(out/"audit_20.csv",audit,columns=acols)
    _rng=random.Random(args.seed)
    allr=[r for sp in("train","validation","test") for r in splits[sp]]
    write_csv(out/"sample_10.csv",_rng.sample(splits['train'],10))
    man={"seed":args.seed,"attempts":attempts,
         "counts":{sp:len(splits[sp]) for sp in splits},"total":len(allr),
         "per_rule":{r:{sp:sum(1 for x in splits[sp] if x["expected_rule_id"]==r) for sp in QUOTAS} for r in RULE_IDS},
         "sha256":{sp:hashlib.sha256((out/f"{sp}.csv").read_bytes()).hexdigest() for sp in("train","validation","test")},
         "checks":validate(splits),"human_audit_completed":False}
    (out/"manifest.json").write_text(json.dumps(man,indent=2))
    print(f"Done. Files in {out}/")

if __name__=="__main__": main()
