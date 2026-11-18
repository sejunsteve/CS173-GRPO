# training.py
import argparse, hashlib, json, os, platform
from pathlib import Path

import torch, datasets, accelerate, peft, transformers, trl
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOTrainer, GRPOConfig

from rewards import access_reward
from dataset import load_rows
from policy import evaluate, make_prompt, canonical, chat_messages


def training_records(path):
    rows = load_rows(path)
    if not rows:
        raise ValueError("No training rows")
    out = []
    for row in rows:
        request = json.loads(row['request_json'])
        if (row['prompt'] != make_prompt(request)
                or json.loads(row['expected_output']) != evaluate(request)):
            raise ValueError(f"Corrupted prompt or label: {row['example_id']}")
        if not row["prompt"].strip():
            raise ValueError(f"Empty prompt: {row['example_id']}")
        if not row.get("expected_decision") or not row.get("expected_rule_id"):
            raise ValueError(f"Missing label: {row['example_id']}")
        out.append({"prompt": chat_messages(row['prompt']),
                    "request_json": canonical(request)})
    return out


def token_length_check(tokenizer, records, limit):
    for i, rec in enumerate(records):
        txt = tokenizer.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
        n = len(tokenizer(txt, add_special_tokens=False).input_ids)
        if n > limit:
            raise ValueError(f"Prompt {i}: {n} tokens > {limit}")


def make_config(output, steps=130, cpu=False, group=4, bf16=False):
    if group < 2 or steps < 1:
        raise ValueError('Need group >= 2 and positive optimizer steps')
    return GRPOConfig(
        output_dir=str(output), max_steps=steps, seed=173, data_seed=173, use_cpu=cpu,
        num_generations=group, per_device_train_batch_size=1,
        gradient_accumulation_steps=group*2, generation_batch_size=group*2,
        max_prompt_length=1536, max_completion_length=48,
        learning_rate=1e-5, temperature=1, top_p=1, top_k=0,
        beta=0.04, epsilon=0.2, num_iterations=1,
        loss_type="grpo", scale_rewards="group",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=False, optim="adamw_torch",
        use_vllm=False, disable_dropout=True, remove_unused_columns=False,
        mask_truncated_completions=True,
        logging_steps=1, save_steps=25, save_total_limit=2,
        report_to="none", push_to_hub=False, dataloader_num_workers=0,
    )


class _StepLog(TrainerCallback):
    def __init__(self, path): self.path = Path(path)
    def on_log(self, args, state, control, logs=None, **kw):
        if state.global_step > 0 and state.is_world_process_zero:
            with open(self.path, "a", encoding='utf-8') as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(prog="training.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--output", default="output/grpo")
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    set_seed(173)

    out = Path(a.output)
    if (out / "COMPLETE").exists() and not a.resume:
        raise SystemExit(f"{out} already completed; pass --resume")
    out.mkdir(parents=True, exist_ok=True)

    if not a.cpu:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required (use --cpu)")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise SystemExit("Single GPU only; WORLD_SIZE>1 unsupported")
        bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
        device = "cuda"
    else:
        bf16, device = False, "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float32

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=False)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type='CAUSAL_LM'))
    # TRL0.24 temporarily disables checkpointing during generation and restores
    # it with the reentrant default. Frozen embeddings must still feed a grad
    # tensor into the checkpointed blocks for adapter gradients to survive.
    model.enable_input_require_grads()

    records = training_records(a.data)
    token_length_check(tok, records, 1536)
    ds = Dataset.from_list(records)
    tlens = [len(tok(tok.apply_chat_template(r["prompt"], tokenize=False,
             add_generation_prompt=True), add_special_tokens=False).input_ids) for r in records]

    cfg = make_config(out, steps=a.steps, cpu=a.cpu, bf16=bf16)
    metrics_path = out / "metrics.jsonl"

    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        reward_funcs=[access_reward],
        processing_class=tok, callbacks=[_StepLog(metrics_path)])

    resume = None
    if a.resume:
        cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
        if cks: resume = str(cks[-1])
    run = {
        "model": a.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "versions": {p: getattr(mod, "__version__", "?") for p, mod in
                     [("torch", torch),
                      ("transformers", transformers), ("peft", peft),
                      ("datasets", datasets), ("accelerate", accelerate),
                      ("trl", trl)]},
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name() if device=='cuda' else 'cpu', "bf16": bf16,
        "config": cfg.to_dict(),
        "prompt_token_range": [min(tlens), max(tlens)],
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str),encoding='utf-8')
    result=trainer.train(resume_from_checkpoint=resume)
    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(adapter_dir)
    trainer.save_state()
    trainer.save_metrics('train',result.metrics)
    (out / "COMPLETE").write_text("Optimization finished; inspect held-out metrics before claiming improvement.\n")
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
