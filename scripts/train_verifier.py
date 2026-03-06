#!/usr/bin/env python
"""
scripts/train_verifier.py
=========================
Fine-tune a verifier model on the Constitutional Q&A dataset (Articles 14–17
of the Indian Constitution) for use as a fairness-aware auditor within the
AMBEDKAR speculative decoding pipeline.

The verifier is trained with a standard causal-LM objective on chat-formatted
constitutional Q&A pairs.  After training it acts as a normative filter:
given two distributions (original prompt vs. counterfactual prompt), the
verifier's per-token log-probabilities are used to compute the JS-divergence
penalty that steers token selection toward identity-invariant continuations.

Reference: §3.3 and Appendix G of the AMBEDKAR paper.

Usage
-----
# Quickstart (GPU recommended)
python scripts/train_verifier.py \\
    --base_model  gpt2 \\
    --data_path   data/constitution_qa/sft_constitution.jsonl \\
    --output_dir  checkpoints/verifier \\
    --epochs      12 \\
    --lr          1e-5 \\
    --batch_size  32

# Larger verifier (e.g. for GPT-2 Large → LLaMA-3.2-3B-Instruct pairing)
python scripts/train_verifier.py \\
    --base_model  gpt2-large \\
    --data_path   data/constitution_qa/sft_constitution.jsonl \\
    --output_dir  checkpoints/verifier-large \\
    --epochs      12 \\
    --lr          5e-6 \\
    --batch_size  16 \\
    --max_length  128
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ConstitutionQADataset(Dataset):
    """
    Reads a JSONL file where each line is either:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    or the flat format:
        {"prompt": "...", "completion": "..."}

    The full turn is concatenated and used as the training target (causal-LM).
    Only the assistant portion contributes to the loss.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load(path)

    def _load(self, path: str):
        examples = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Line %d: JSON parse error — %s", lineno, exc)
                    continue

                if "messages" in obj:
                    user = next(
                        (m["content"] for m in obj["messages"] if m["role"] == "user"), ""
                    )
                    assistant = next(
                        (m["content"] for m in obj["messages"] if m["role"] == "assistant"), ""
                    )
                elif "prompt" in obj and "completion" in obj:
                    user = obj["prompt"]
                    assistant = obj["completion"]
                else:
                    log.warning("Line %d: unrecognised format, skipping.", lineno)
                    continue

                examples.append({"user": user.strip(), "assistant": assistant.strip()})

        log.info("Loaded %d examples from %s", len(examples), path)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # Format: "<user turn>\n<assistant turn><EOS>"
        user_text = f"Question: {ex['user']}\nAnswer: "
        full_text = user_text + ex["assistant"] + self.tokenizer.eos_token

        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        user_enc = self.tokenizer(
            user_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Labels: -100 on the user portion so only assistant tokens contribute
        labels = input_ids.clone()
        user_len = user_enc["input_ids"].shape[1]
        labels[:user_len] = -100
        labels[attention_mask == 0] = -100  # also mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    log.info("Using device: %s", device)

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    log.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32,
    )
    model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ConstitutionQADataset(args.data_path, tokenizer, args.max_length)

    # Train/val split (80/20)
    val_size = max(1, int(len(dataset) * 0.20))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        log.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch, args.epochs, avg_train_loss, avg_val_loss,
        )

        # Early stopping + checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            log.info("  ✓ New best — saving checkpoint to %s", output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            # Save training args alongside checkpoint
            with open(output_dir / "training_args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
        else:
            patience_counter += 1
            log.info("  No improvement (%d/%d patience)", patience_counter, args.patience)
            if patience_counter >= args.patience:
                log.info("Early stopping triggered at epoch %d.", epoch)
                break

    log.info("Training complete. Best val loss: %.4f", best_val_loss)
    log.info("Verifier saved to: %s", output_dir)
    log.info(
        "Load it with:\n  AMBEDKARDecoder.from_pretrained(\n"
        "      draft_model_name='gpt2',\n"
        "      verifier_model_name='%s',\n  ...)\n",
        str(output_dir),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune the AMBEDKAR verifier on Constitutional Q&A data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model", default="gpt2",
                   help="HuggingFace model ID or local path for the verifier base.")
    p.add_argument("--data_path", default="data/constitution_qa/sft_constitution.jsonl",
                   help="Path to the Constitutional Q&A JSONL file.")
    p.add_argument("--output_dir", default="checkpoints/verifier",
                   help="Directory to save the trained verifier checkpoint.")
    p.add_argument("--epochs", type=int, default=12,
                   help="Maximum training epochs. Early stopping may terminate sooner.")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size per training step.")
    p.add_argument("--max_length", type=int, default=128,
                   help="Maximum token length for Q&A pairs.")
    p.add_argument("--patience", type=int, default=2,
                   help="Early stopping patience (epochs without val improvement).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--fp16", action="store_true",
                   help="Enable mixed-precision (FP16) training on CUDA.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available (useful for debugging).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    log.info("AMBEDKAR Verifier Training")
    log.info("  base_model : %s", args.base_model)
    log.info("  data_path  : %s", args.data_path)
    log.info("  output_dir : %s", args.output_dir)
    log.info("  epochs     : %d  lr=%.2e  batch=%d", args.epochs, args.lr, args.batch_size)
    train(args)


if __name__ == "__main__":
    main()
