"""
ambedkar.core.decoding
======================
Fairness-Aware Speculative Decoding — the main inference engine of AMBEDKAR.

Pipeline (Figure 5 / Algorithm 1 of the paper):
    Promptor → Speculativa → Contrarium → Aequitas → Moderatus

Reference: §3 "AMBEDKAR: Debiasing with Fairness-Aware Speculative Decoding"
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .contrarium import Contrarium
from ..utils.divergence import js_divergence_scalars

logger = logging.getLogger(__name__)


@dataclass
class AMBEDKARConfig:
    """
    Configuration for the AMBEDKAR fairness-aware speculative decoder.

    Parameters
    ----------
    alpha : float
        Fairness-fluency trade-off coefficient α in the paper objective:
            c* = argmax [log p(c|x,y<t) - α · D_t(c)]
        Higher α → stronger counterfactual stability, potentially lower fluency.
        Recommended range: 0.5–2.0. Paper default: 1.0 (Appendix G).
    top_k : int
        Speculative candidate set size k. Paper uses k=5 (Appendix G).
    max_new_tokens : int
        Maximum generation length in tokens.
    divergence : str
        Divergence used by Aequitas. One of {"js", "kl", "fast"}.
        "js" (Jensen-Shannon) is bounded, symmetric, and best-performing (Table 3).
    device : str
        Torch device string ("cuda", "cpu", "mps").
    fp16 : bool
        Load models in half precision (recommended for GPU deployment).
    seed : int
        RNG seed for reproducibility.
    draft_temperature : float
        Sampling temperature for draft model (τ=0.7, Appendix G).
    verifier_temperature : float
        Temperature for verifier scoring (τ=0 → greedy, Appendix G).
    """
    alpha: float = 1.0
    top_k: int = 5
    max_new_tokens: int = 80
    divergence: str = "js"
    device: str = "cpu"
    fp16: bool = False
    seed: int = 42
    draft_temperature: float = 0.7
    verifier_temperature: float = 0.0

    def __post_init__(self):
        if self.divergence not in ("js", "kl", "fast"):
            raise ValueError(f"divergence must be 'js','kl', or 'fast'; got '{self.divergence}'")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


class AMBEDKARDecoder:
    """
    AMBEDKAR: Fairness-Aware Speculative Decoding.

    At each decoding step t, given prompt x and prefix y_{<t}:

      1. **Speculativa** — draft model proposes top-k candidate tokens C_t.
      2. **Contrarium**  — constructs counterfactual prompt x'.
      3. **Aequitas**    — verifier scores each c in C_t via D_JS.
      4. **Moderatus**   — selects:
             c* = argmax_{c in C_t} [log p(c|x,y_{<t}) - α·D_t(c)]

    Draft and verifier can be **heterogeneous** (different model families,
    tokenizers, sizes) — a key design feature enabling black-box deployment.

    Parameters
    ----------
    draft_model, draft_tokenizer
        The proposer (SLM); proposes speculative candidates.
    verifier_model, verifier_tokenizer
        The fairness auditor (constitutionally fine-tuned LLM).
    config : AMBEDKARConfig
    contrarium : Contrarium, optional
        If None, a default Contrarium instance is created.

    Examples
    --------
    Quick-start (white-box models from HuggingFace Hub):

    >>> from ambedkar import AMBEDKARDecoder, AMBEDKARConfig
    >>> cfg = AMBEDKARConfig(alpha=1.0, top_k=5, max_new_tokens=60)
    >>> decoder = AMBEDKARDecoder.from_pretrained("gpt2", "gpt2-large", config=cfg)
    >>> out = decoder.generate("As a [MASK] applying for a leadership role, I feel")
    >>> print(out)

    With metadata (divergence traces per step):

    >>> out, meta = decoder.generate(prompt, return_metadata=True)
    >>> print(f"Mean divergence: {meta['mean_divergence']:.4f}")
    """

    def __init__(
        self,
        draft_model,
        draft_tokenizer,
        verifier_model,
        verifier_tokenizer,
        config: Optional[AMBEDKARConfig] = None,
        contrarium: Optional[Contrarium] = None,
    ):
        self.cfg = config or AMBEDKARConfig()
        self.draft_model = draft_model.eval()
        self.draft_tokenizer = draft_tokenizer
        self.verifier_model = verifier_model.eval()
        self.verifier_tokenizer = verifier_tokenizer
        self.contrarium = contrarium or Contrarium()

        for tok in (self.draft_tokenizer, self.verifier_tokenizer):
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        logger.info(
            "AMBEDKARDecoder ready | α=%.2f | k=%d | div=%s | device=%s",
            self.cfg.alpha, self.cfg.top_k, self.cfg.divergence, self.cfg.device,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        draft_model_name: str,
        verifier_model_name: str,
        config: Optional[AMBEDKARConfig] = None,
        contrarium: Optional[Contrarium] = None,
        **hf_kwargs,
    ) -> "AMBEDKARDecoder":
        """
        Load both models directly from HuggingFace Hub.

        Parameters
        ----------
        draft_model_name : str
            HuggingFace model id or local path for the draft model.
        verifier_model_name : str
            HuggingFace model id or local path for the verifier.
            Typically a model fine-tuned on Constitutional Q&A.
        config : AMBEDKARConfig, optional
        contrarium : Contrarium, optional
        **hf_kwargs :
            Forwarded to ``AutoModelForCausalLM.from_pretrained``
            (e.g., ``cache_dir``, ``trust_remote_code``).

        Examples
        --------
        >>> # Homogeneous pair (paper Table 5)
        >>> decoder = AMBEDKARDecoder.from_pretrained("gpt2", "gpt2-large")

        >>> # Heterogeneous pair (paper Table 5)
        >>> decoder = AMBEDKARDecoder.from_pretrained(
        ...     "sarvamai/OpenHathi-7B-Hi-v0.1",
        ...     "openai-community/gpt2-xl",
        ...     config=AMBEDKARConfig(alpha=1.0, device="cuda"),
        ... )
        """
        cfg = config or AMBEDKARConfig()
        dtype = torch.float16 if cfg.fp16 else torch.float32

        logger.info("Loading draft model: %s", draft_model_name)
        d_tok = AutoTokenizer.from_pretrained(draft_model_name, **hf_kwargs)
        d_mdl = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=dtype, **hf_kwargs
        ).to(cfg.device)

        logger.info("Loading verifier model: %s", verifier_model_name)
        v_tok = AutoTokenizer.from_pretrained(verifier_model_name, **hf_kwargs)
        v_mdl = AutoModelForCausalLM.from_pretrained(
            verifier_model_name, torch_dtype=dtype, **hf_kwargs
        ).to(cfg.device)

        return cls(d_mdl, d_tok, v_mdl, v_tok, cfg, contrarium)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_metadata: bool = False,
    ):
        """
        Generate a fairness-aware continuation for *prompt*.

        Parameters
        ----------
        prompt : str
            Input text. May contain ``[MASK]`` for identity-inference probing.
        max_new_tokens : int, optional
            Overrides ``config.max_new_tokens`` for this call only.
        return_metadata : bool
            When True, returns ``(continuation, metadata_dict)`` where
            ``metadata_dict`` includes per-step divergence scores and latency.

        Returns
        -------
        str
            Generated continuation (prompt prefix is stripped).
        tuple (str, dict), only when return_metadata=True
            ``(continuation, {"steps": [...], "total_latency_s": float,
               "mean_divergence": float, "counterfactual_prompt": str})``
        """
        n_tokens = max_new_tokens or self.cfg.max_new_tokens
        generated = prompt
        meta_steps: List[Dict] = []
        t0 = time.perf_counter()

        # Contrarium: build counterfactual once per call
        cf_prompt = self.contrarium.perturb(prompt)

        for step in range(n_tokens):
            # Speculativa
            candidates, log_prob_map = self._propose_candidates(generated)
            if not candidates:
                break

            # Aequitas
            divergences = self._score_candidates(candidates, generated, cf_prompt)

            # Moderatus
            best_token, best_score = self._select_token(candidates, log_prob_map, divergences)

            token_str = self.draft_tokenizer.convert_tokens_to_string([best_token])
            generated += token_str

            if return_metadata:
                meta_steps.append({
                    "step": step,
                    "token": best_token,
                    "score": best_score,
                    "divergence": divergences.get(best_token, 0.0),
                })

            best_id = self.draft_tokenizer.convert_tokens_to_ids(best_token)
            if best_id == self.draft_tokenizer.eos_token_id:
                break

        continuation = generated[len(prompt):]
        total_latency = time.perf_counter() - t0

        if return_metadata:
            mean_div = float(np.mean([s["divergence"] for s in meta_steps])) if meta_steps else 0.0
            return continuation, {
                "steps": meta_steps,
                "total_latency_s": total_latency,
                "mean_divergence": mean_div,
                "counterfactual_prompt": cf_prompt,
            }
        return continuation

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate continuations for multiple prompts sequentially."""
        return [self.generate(p, **kwargs) for p in prompts]

    # ------------------------------------------------------------------
    # Pipeline stages (internal)
    # ------------------------------------------------------------------

    def _propose_candidates(self, prefix: str) -> Tuple[List[str], Dict[str, float]]:
        """Speculativa: draft model → top-k token strings + log-probs."""
        inputs = self.draft_tokenizer(
            prefix, return_tensors="pt", truncation=True, max_length=512
        ).to(self.cfg.device)
        with torch.no_grad():
            outputs = self.draft_model(**inputs)
        log_p = outputs.logits[0, -1].log_softmax(dim=-1)
        topk = torch.topk(log_p, self.cfg.top_k)
        ids = topk.indices.tolist()
        tokens = self.draft_tokenizer.convert_ids_to_tokens(ids)
        lp_map = {t: float(lp) for t, lp in zip(tokens, topk.values.tolist())}
        return tokens, lp_map

    def _score_candidates(
        self, candidates: List[str], original: str, counterfactual: str
    ) -> Dict[str, float]:
        """Aequitas: compute divergence D_t(c) for each candidate c."""
        orig_probs = self._verifier_probs(original, candidates)
        cf_probs = self._verifier_probs(counterfactual, candidates)
        return js_divergence_scalars(orig_probs, cf_probs, metric=self.cfg.divergence)

    def _verifier_probs(self, prefix: str, candidates: List[str]) -> Dict[str, float]:
        """Run verifier, return probability for each candidate token string."""
        inputs = self.verifier_tokenizer(
            prefix, return_tensors="pt", truncation=True, max_length=512
        ).to(self.cfg.device)
        with torch.no_grad():
            outputs = self.verifier_model(**inputs)
        probs = outputs.logits[0, -1].softmax(dim=-1)
        result: Dict[str, float] = {}
        for token in candidates:
            vid = self.verifier_tokenizer.convert_tokens_to_ids(token)
            if vid is not None and 0 <= vid < probs.shape[0]:
                result[token] = float(probs[vid])
            else:
                result[token] = 0.0
        return result

    def _select_token(
        self,
        candidates: List[str],
        log_probs: Dict[str, float],
        divergences: Dict[str, float],
    ) -> Tuple[str, float]:
        """Moderatus: select c* = argmax [log p(c) - α · D(c)]."""
        best_token, best_score = candidates[0], float("-inf")
        for c in candidates:
            score = log_probs.get(c, -1e9) - self.cfg.alpha * divergences.get(c, 0.0)
            if score > best_score:
                best_score, best_token = score, c
        return best_token, best_score
