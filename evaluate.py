"""Greedy before/after evaluation; validation by default, test only on request."""
import argparse
import json
from pathlib import Path
from dataset import load_rows
from rewards import parse_output, score


def metrics(records):
    if not records:
        raise ValueError('No evaluation records')
    n = len(records)
    result = {'n': n, 'average_reward': sum(r['reward'] for r in records) / n,
              'exact_match': sum(r['reward'] == 1.0 for r in records) / n,
              'decision_accuracy': sum(r['decision_correct'] for r in records) / n,
              'valid_json': sum(r['valid_json'] for r in records) / n}
    result['per_rule'] = {}
    for rule in sorted({r['expected']['rule_id'] for r in records}):
        group = [r for r in records if r['expected']['rule_id'] == rule]
        result['per_rule'][rule] = {'n': len(group), 'exact_match': sum(r['reward'] == 1.0 for r in group) / len(group)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--revision', default=None)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--data', type=Path, default=Path('data/validation.csv'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from policy import evaluate, make_prompt, chat_messages
    set_seed(173)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('Evaluation requires a GPU allocation or explicit --cpu')
    if args.adapter:
        run_path = args.adapter.parent / 'run.json'
        if run_path.exists():
            run = json.loads(run_path.read_text(encoding='utf-8'))
            args.model = run['model']
            args.revision = run['model_revision']
    device = 'cpu' if args.cpu else 'cuda'
    bf16 = device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype,
                                                attn_implementation='sdpa', trust_remote_code=False).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = load_rows(args.data)
    if args.limit:
        # Deterministic per-rule cycling avoids judging a single-rule prefix.
        groups = {rule: [r for r in rows if r['expected_rule_id'] == rule] for rule in sorted({r['expected_rule_id'] for r in rows})}
        rows = [r for layer in zip(*groups.values()) for r in layer][:args.limit]
    records = []
    for row in rows:
        request = json.loads(row['request_json'])
        if row['prompt'] != make_prompt(request):
            raise ValueError('Evaluation prompt mismatch')
        inputs = tokenizer.apply_chat_template(chat_messages(row['prompt']),
                    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=48,
                                       pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        output = parse_output(text)
        expected = evaluate(request)
        records.append({'example_id': row['example_id'], 'expected': expected, 'completion': text,
                        'reward': score(text, request), 'valid_json': output is not None,
                        'decision_correct': output is not None and output['decision'] == expected['decision']})
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'model': args.model, 'model_revision': getattr(model.config, '_commit_hash', None),
              'adapter': str(args.adapter) if args.adapter else None, 'data': str(args.data), **metrics(records)}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
