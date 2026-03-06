"""
AMBEDKAR — Adaptive Mitigation of Bias through Equitable Decoding
              and Knowledge-Aware Re-ranking

A training-free, inference-time debiasing framework that leverages
fairness-aware speculative decoding with a constitutionally-grounded
verifier to reduce identity-conditioned bias in LLM outputs.

Paper: "AMBEDKAR: Adaptive Mitigation of Bias through Equitable
        Decoding and Knowledge-Aware Re-ranking" (ACL 2026)
Benchmark: AI Constitution of India — https://www.aiconstitutionofindia.in

Quick-start
-----------
>>> from ambedkar import AMBEDKARDecoder, AMBEDKARConfig
>>> cfg = AMBEDKARConfig(alpha=1.0, top_k=5, max_new_tokens=80)
>>> decoder = AMBEDKARDecoder.from_pretrained(
...     draft_model_name="gpt2",
...     verifier_model_name="gpt2-large",
...     config=cfg,
... )
>>> output = decoder.generate("As a [MASK] applying for a leadership role, I feel")
>>> print(output)
"""

from ambedkar.core.decoding import AMBEDKARDecoder, AMBEDKARConfig
from ambedkar.core.contrarium import Contrarium
from ambedkar.evaluation.iir import IIREvaluator, IIRResult
from ambedkar.utils.divergence import js_divergence_distributions, js_divergence_scalars

__version__ = "1.0.0"
__author__ = "Anonymous (ACL 2026 Submission)"
__license__ = "Apache-2.0"

__all__ = [
    # Core pipeline
    "AMBEDKARDecoder",
    "AMBEDKARConfig",
    "Contrarium",
    # Evaluation
    "IIREvaluator",
    "IIRResult",
    # Divergence utilities
    "js_divergence_distributions",
    "js_divergence_scalars",
]
