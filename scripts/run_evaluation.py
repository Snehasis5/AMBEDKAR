#!/usr/bin/env python
"""
scripts/run_evaluation.py
=========================
End-to-end evaluation of AMBEDKAR on the AI Constitution of India benchmark.

Runs the full 5-stage pipeline (Promptor → Speculativa → Contrarium →
Aequitas → Moderatus) over a JSONL prompt file and reports:
  • Overall IIR (Identity Inference Rate)
  • Per-religion IIR breakdown
  • Per-caste IIR breakdown (if caste prompts supplied)
  • AMBEDKAR vs. baseline comparison
  • Latency statistics (mean / p50 / p95 / p99)

Results are written to JSON and a human-readable Markdown table.

Reference: §4 and Appendix G of the AMBEDKAR paper.

Usage
-----
# Evaluate baseline GPT-2 vs AMBEDKAR (GPT-2 + GPT-2-Large verifier)
python scripts/run_evaluation.py \\
    --draft_model   gpt2 \\
    --verifier_model gpt2-large \\
    --prompts       data/sample_prompts/religion_prompts.jsonl \\
    --axis          religion \\
    --output_dir    results/religion

# Evaluate heterogeneous pairing (paper Table 5)
python scripts/run_evaluation.py \\
    --draft_model   gpt2-large \\
    --verifier_model meta-llama/Llama-3.2-3B-Instruct \\
    --prompts       data/sample_prompts/caste_prompts.jsonl \\
    --axis          caste \\
    --alpha         1.0 \\
    --top_k         5 \\
    --output_dir    results/caste-heterogeneous

# Baseline-only run (no AMBEDKAR, just IIR measurement)
python scripts/run_evaluation.py \\
    --draft_model gpt2 \\
    --prompts     data/sample_prompts/religion_prompts.jsonl \\
    --baseline_only
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ambedkar import AMBEDKARConfig, AMBEDKARDecoder, IIREvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list[dict]:
    """Load JSONL prompts. Each line: {"prompt": "...", "gold_identity": "..."}"""
    prompts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    log.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def percentile(arr: list[float], p: float) -> float:
    return float(np.percentile(arr, p))


def format_markdown_table(results: dict[str, Any]) -> str:
    lines = [
        "## AMBEDKAR Evaluation Results",
        "",
        f"**Draft model**: `{results['draft_model']}`  ",
        f"**Verifier model**: `{results['verifier_model']}`  ",
        f"**α**: {results['alpha']}  **top-k**: {results['top_k']}  ",
        f"**Prompts evaluated**: {results['n_prompts']}",
        "",
        "### IIR Summary",
        "",
        "| Condition | IIR (↓ better) |",
        "|-----------|---------------|",
        f"| Baseline (no AMBEDKAR) | {results['baseline_iir']:.3f} |",
        f"| AMBEDKAR | {results['ambedkar_iir']:.3f} |",
        f"| Relative reduction | **{results['relative_reduction_pct']:.1f}%** |",
        f"| Absolute reduction | {results['absolute_reduction']:.3f} |",
        "",
        "### Latency",
        "",
        "| Statistic | Baseline (s/tok) | AMBEDKAR (s/tok) | Overhead |",
        "|-----------|-----------------|-----------------|---------|",
    ]
    bl = results["baseline_latency"]
    am = results["ambedkar_latency"]
    for stat in ["mean", "p50", "p95", "p99"]:
        overhead = (am[stat] / bl[stat] - 1) * 100 if bl[stat] > 0 else float("nan")
        lines.append(
            f"| {stat} | {bl[stat]:.4f} | {am[stat]:.4f} | {overhead:+.1f}% |"
        )
    lines.append("")

    if results.get("per_group_iir"):
        lines += ["### Per-Group IIR", "", "| Group | Baseline IIR | AMBEDKAR IIR | Δ |",
                  "|-------|-------------|-------------|---|"]
        for group, vals in sorted(results["per_group_iir"].items()):
            delta = vals["ambedkar"] - vals["baseline"]
            lines.append(
                f"| {group} | {vals['baseline']:.3f} | {vals['ambedkar']:.3f} | {delta:+.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
        log.info("Capped at %d prompts (--max_prompts).", args.max_prompts)

    evaluator = IIREvaluator()

    # ── Baseline: greedy decode with draft model only ─────────────────────────
    log.info("Running BASELINE evaluation …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    bl_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if bl_tokenizer.pad_token is None:
        bl_tokenizer.pad_token = bl_tokenizer.eos_token
    bl_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)
    bl_model.eval()

    baseline_outputs, baseline_latencies = [], []
    for item in prompts:
        prompt_text = item["prompt"]
        t0 = time.perf_counter()
        with torch.no_grad():
            enc = bl_tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            out = bl_model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=bl_tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        n_new_tokens = out.shape[1] - enc["input_ids"].shape[1]
        per_tok = elapsed / max(n_new_tokens, 1)
        baseline_latencies.append(per_tok)
        generated = bl_tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        baseline_outputs.append({"prompt": prompt_text, "generated": generated})

    del bl_model  # free memory before loading AMBEDKAR

    baseline_result = evaluator.evaluate(baseline_outputs)
    log.info("Baseline IIR = %.4f", baseline_result.overall_iir)

    if args.baseline_only:
        report = {
            "draft_model": args.draft_model,
            "verifier_model": "N/A (baseline only)",
            "alpha": "N/A",
            "top_k": "N/A",
            "n_prompts": len(prompts),
            "baseline_iir": baseline_result.overall_iir,
            "per_group_iir_baseline": baseline_result.per_religion_iir,
        }
        with open(output_dir / "baseline_results.json", "w") as f:
            json.dump(report, f, indent=2)
        log.info("Baseline results saved to %s", output_dir / "baseline_results.json")
        return

    # ── AMBEDKAR evaluation ───────────────────────────────────────────────────
    log.info("Running AMBEDKAR evaluation …")
    cfg = AMBEDKARConfig(
        alpha=args.alpha,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        device=str(device),
        fp16=(not args.cpu and torch.cuda.is_available()),
        seed=args.seed,
    )
    decoder = AMBEDKARDecoder.from_pretrained(
        draft_model_name=args.draft_model,
        verifier_model_name=args.verifier_model,
        config=cfg,
    )

    ambedkar_outputs, ambedkar_latencies = [], []
    for i, item in enumerate(prompts, 1):
        prompt_text = item["prompt"]
        t0 = time.perf_counter()
        generated = decoder.generate(prompt_text)
        elapsed = time.perf_counter() - t0

        # crude per-token estimate
        n_toks = len(generated.split())
        ambedkar_latencies.append(elapsed / max(n_toks, 1))
        ambedkar_outputs.append({"prompt": prompt_text, "generated": generated})

        if i % 10 == 0:
            log.info("  %d/%d prompts processed …", i, len(prompts))

    ambedkar_result = evaluator.evaluate(ambedkar_outputs)
    log.info("AMBEDKAR IIR = %.4f", ambedkar_result.overall_iir)

    # ── Per-group breakdown ───────────────────────────────────────────────────
    per_group: dict[str, dict] = {}
    bl_groups = baseline_result.per_religion_iir or {}
    am_groups = ambedkar_result.per_religion_iir or {}
    for g in set(list(bl_groups.keys()) + list(am_groups.keys())):
        per_group[g] = {
            "baseline": bl_groups.get(g, float("nan")),
            "ambedkar": am_groups.get(g, float("nan")),
        }

    # ── Latency stats ─────────────────────────────────────────────────────────
    def latency_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "p50": percentile(arr, 50),
            "p95": percentile(arr, 95),
            "p99": percentile(arr, 99),
        }

    bl_iir = baseline_result.overall_iir
    am_iir = ambedkar_result.overall_iir
    abs_red = bl_iir - am_iir
    rel_red = (abs_red / bl_iir * 100) if bl_iir > 0 else 0.0

    results = {
        "draft_model": args.draft_model,
        "verifier_model": args.verifier_model,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "n_prompts": len(prompts),
        "baseline_iir": bl_iir,
        "ambedkar_iir": am_iir,
        "absolute_reduction": abs_red,
        "relative_reduction_pct": rel_red,
        "baseline_latency": latency_stats(baseline_latencies),
        "ambedkar_latency": latency_stats(ambedkar_latencies),
        "per_group_iir": per_group,
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    md = format_markdown_table(results)
    with open(output_dir / "results.md", "w") as f:
        f.write(md)

    # Save raw outputs for downstream analysis
    with open(output_dir / "baseline_outputs.jsonl", "w") as f:
        for o in baseline_outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(output_dir / "ambedkar_outputs.jsonl", "w") as f:
        for o in ambedkar_outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    log.info("=" * 60)
    log.info("Baseline IIR : %.4f", bl_iir)
    log.info("AMBEDKAR IIR : %.4f", am_iir)
    log.info("Abs. reduction: %.4f  (%.1f%% relative)", abs_red, rel_red)
    log.info("Results saved to: %s", output_dir)
    print(md)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end AMBEDKAR evaluation on AI Constitution of India prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--draft_model", default="gpt2",
                   help="HuggingFace model ID or local path for the draft (proposer) model.")
    p.add_argument("--verifier_model", default="gpt2-large",
                   help="HuggingFace model ID or local path for the verifier model.")
    p.add_argument("--prompts", default="data/sample_prompts/religion_prompts.jsonl",
                   help="Path to evaluation prompts (JSONL).")
    p.add_argument("--axis", choices=["religion", "caste", "both"], default="religion",
                   help="Bias axis for per-group IIR breakdown.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Fairness–fluency trade-off coefficient α (Eq. 1 in paper).")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of speculative candidates k.")
    p.add_argument("--max_new_tokens", type=int, default=80,
                   help="Maximum new tokens to generate per prompt.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap the number of evaluated prompts (for quick debug runs).")
    p.add_argument("--output_dir", default="results",
                   help="Directory to write results JSON, Markdown, and JSONL outputs.")
    p.add_argument("--baseline_only", action="store_true",
                   help="Only run baseline (no AMBEDKAR). Useful for IIR measurement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference (slow but useful for debugging).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
