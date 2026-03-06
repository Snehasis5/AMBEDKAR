"""
ambedkar.utils.divergence
=========================
Divergence utilities for the Aequitas fairness-scoring stage.

Implements the Jensen-Shannon (JS) divergence used in the AMBEDKAR objective:

    D_t(c) = D_JS[ p(·|x, y<t⊕c) , p(·|x', y<t⊕c) ]

where D_JS is bounded in [0, log 2], symmetric, and numerically stable —
the recommended default per Table 3 of the paper.

Also provides KL divergence and a fast approximation baseline for ablations.

Reference: §3.1 Principle 3, Table 3, Appendix K "Divergence Sensitivity Analysis"
"""
from __future__ import annotations

import numpy as np
from typing import Dict


_EPS = 1e-10  # numerical stability floor


# ---------------------------------------------------------------------------
# Distribution-level (full-vocab) divergences
# Used during evaluation / analysis; not in the hot decoding path.
# ---------------------------------------------------------------------------

def js_divergence_distributions(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability vectors.

    D_JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M),   M = (P+Q)/2

    Bounded in [0, log 2] ≈ [0, 0.693] (base-e) or [0, 1] (base-2).
    Symmetric: D_JS(P, Q) = D_JS(Q, P).

    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors (will be renormalised if they don't sum to 1).

    Returns
    -------
    float in [0, log 2]
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, _EPS, None); p /= p.sum()
    q = np.clip(q, _EPS, None); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def kl_divergence_distributions(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL divergence KL(P||Q).  Asymmetric; can be infinite.

    Parameters
    ----------
    p, q : np.ndarray   reference and approximation distributions.

    Returns
    -------
    float >= 0
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, _EPS, None); p /= p.sum()
    q = np.clip(q, _EPS, None); q /= q.sum()
    return float(_kl(p, q))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation distance: TV(P, Q) = 0.5 * sum|P_i - Q_i|.
    Bounded in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, _EPS, None); p /= p.sum()
    q = np.clip(q, _EPS, None); q /= q.sum()
    return float(0.5 * np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# Scalar-level (per-candidate-token) divergences
# Hot path: called once per decoding step with k candidate scores.
# ---------------------------------------------------------------------------

def js_divergence_scalars(
    orig_probs: Dict[str, float],
    cf_probs: Dict[str, float],
    metric: str = "js",
) -> Dict[str, float]:
    """
    Compute per-token divergence scores for Aequitas stage.

    For each candidate token c, this computes a scalar D(c) measuring
    how much the verifier's probability of c changes between the original
    and counterfactual prompts.

    This is the token-local approximation of D_t(c) used in Algorithm 1.

    Parameters
    ----------
    orig_probs : dict
        {token_str: p(c | x, y_{<t})} from verifier on original prompt.
    cf_probs : dict
        {token_str: p(c | x', y_{<t})} from verifier on counterfactual.
    metric : str
        One of "js" (default), "kl", "fast".
        - "js"   : symmetric JS divergence per-scalar pair.
        - "kl"   : forward KL(orig || cf).
        - "fast" : absolute difference |p_orig - p_cf|  (ablation baseline).

    Returns
    -------
    dict
        {token_str: divergence_score (float >= 0)}

    Notes
    -----
    JS per-scalar pair treats each (p, q) as a Bernoulli(p)/Bernoulli(q)
    distribution.  Full distribution-level JS is used only in evaluation.
    """
    results: Dict[str, float] = {}
    for token in orig_probs:
        p = max(float(orig_probs.get(token, 0.0)), _EPS)
        q = max(float(cf_probs.get(token, 0.0)), _EPS)

        if metric == "js":
            results[token] = _js_scalar(p, q)
        elif metric == "kl":
            results[token] = _kl_scalar(p, q)
        elif metric == "fast":
            results[token] = abs(p - q)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q) with epsilon-clipped Q."""
    q = np.clip(q, _EPS, None)
    return float(np.sum(p * np.log(p / q), where=(p > 0), initial=0.0))


def _js_scalar(p: float, q: float) -> float:
    """JS divergence for scalar Bernoulli pair (p, q)."""
    m = 0.5 * (p + q)
    kl_pm = p * np.log(p / m) if p > 0 else 0.0
    kl_qm = q * np.log(q / m) if q > 0 else 0.0
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _kl_scalar(p: float, q: float) -> float:
    """KL(p||q) for scalars."""
    if p <= 0:
        return 0.0
    q = max(q, _EPS)
    return float(p * np.log(p / q))
