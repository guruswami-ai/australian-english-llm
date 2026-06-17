#!/usr/bin/env python3
"""
v0: LoRA continued-pretraining of Gemma-4-E4B on the Australian-English corpus.
Runs on hog11 (RTX 5090). Consumes PRE-TOKENISED packed sequences (input_ids,
2048 tokens each) — no tokenisation at train time.

  ./.venv/bin/python train_cpt_lora.py --train data/au_cpt_train_gemma4_2048_200k.parquet

Produces a LoRA adapter for fast evaluation of whether the AU register sticks.
This is the cheap iteration run; the production model is a full fine-tune.
"""
import argparse, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen3-8B"   # dense base: bnb 4-bit + PEFT work cleanly (no MatFormer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/au_cpt_train_gemma4_2048_200k.parquet")
    ap.add_argument("--val", default="data/au_cpt_val_gemma4_2048.parquet")
    ap.add_argument("--out", default="models/aussie-gemma4-e4b-lora-v0")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--base", default=MODEL, help="base model id (use Qwen3-8B-Base for v1)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--maxlen", type=int, default=2048, help="slice packed seqs to fit VRAM")
    ap.add_argument("--max_steps", type=int, default=-1, help="cap steps for a v0 (-1 = full epochs)")
    ap.add_argument("--eval_n", type=int, default=256, help="eval on a small held-out subset (full set is too slow)")
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files={"train": args.train, "validation": args.val})
    print({k: len(v) for k, v in ds.items()})

    # QLoRA: 4-bit dense base (~5GB) fits the 5090 comfortably with 2048 ctx.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb_config, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ml = args.maxlen
    def collate(batch):
        ids = torch.tensor([b["input_ids"][:ml] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(),
                "attention_mask": torch.ones_like(ids)}

    eval_ds = ds["validation"].select(range(min(args.eval_n, len(ds["validation"]))))
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=1,  # 151k-vocab CE OOMs at the default batch 8
        learning_rate=args.lr, bf16=True, logging_steps=20, warmup_steps=100,
        lr_scheduler_type="cosine", save_strategy="steps", save_steps=500,
        eval_strategy="steps", eval_steps=250,
        report_to="none", dataloader_num_workers=4, save_total_limit=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=eval_ds, data_collator=collate)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("LoRA adapter ->", args.out)


if __name__ == "__main__":
    main()
