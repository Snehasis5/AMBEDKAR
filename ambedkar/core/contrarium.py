"""
ambedkar.core.contrarium
========================
CONTRARIUM: Adversarial Counterfactual Prompt Constructor.

Implements §3.1 Principle 2 and Appendix G "Generating Counterfactuals".

For a prompt x = [w_1, ..., w_n], Contrarium constructs x' by minimally
perturbing contextually salient, NON-identity tokens via a swap dictionary S:

    w'_i = S(w_i)  if w_i in V_contrast
    w'_i = w_i     otherwise

Identity terms (e.g. "[MASK]", caste names, religion terms) are intentionally
left unchanged to prevent identity leakage into the counterfactual.

The resulting pair (x, x') differs only in non-identity framing
(e.g. "violent" → "peaceful"), forcing the verifier to measure whether
token distributions remain stable under framing shifts — an actionable proxy
for identity-prior amplification (§3.1, Principle 3).

Antonym seeds from WordNet 3.1 (Miller, 1995), supplemented with curated pairs.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default swap dictionary (Vcontrast)
# Curated pairs from WordNet antonyms + paper-specific thesaurus (Appendix G)
# Identity terms (religion/caste) are NEVER included as swap targets.
# ---------------------------------------------------------------------------
CONTRARIUM_DEFAULT_SWAPS: Dict[str, str] = {
    # Violence / safety
    "violent": "peaceful",      "peaceful": "violent",
    "aggressive": "calm",       "calm": "aggressive",
    "dangerous": "safe",        "safe": "dangerous",
    "harmful": "beneficial",    "beneficial": "harmful",
    "threatening": "reassuring","brutal": "gentle",
    "hostile": "friendly",      "friendly": "hostile",
    "criminal": "law-abiding",  "corrupt": "honest",
    "terrorist": "activist",    "extremist": "moderate",
    "radical": "moderate",

    # Education / capability
    "educated": "uneducated",   "uneducated": "educated",
    "literate": "illiterate",   "illiterate": "literate",
    "intelligent": "unintelligent",
    "capable": "incapable",     "incapable": "capable",
    "skilled": "unskilled",     "unskilled": "skilled",
    "qualified": "unqualified", "competent": "incompetent",
    "talented": "untalented",   "lazy": "hardworking",
    "hardworking": "lazy",      "diligent": "careless",

    # Social status / privilege
    "privileged": "marginalized","marginalized": "privileged",
    "wealthy": "poor",          "poor": "wealthy",
    "rich": "impoverished",     "prosperous": "struggling",
    "powerful": "powerless",    "dominant": "subordinate",
    "superior": "inferior",     "inferior": "superior",
    "elite": "common",          "noble": "common",
    "respected": "disrespected","honored": "dishonored",

    # Moral / ethical framing
    "honest": "dishonest",      "dishonest": "honest",
    "trustworthy": "untrustworthy",
    "moral": "immoral",         "immoral": "moral",
    "righteous": "corrupt",     "good": "bad",
    "bad": "good",              "evil": "virtuous",
    "virtuous": "evil",

    # Religious context (non-identity — these are descriptive words)
    "devout": "irreligious",    "pious": "impious",
    "orthodox": "progressive",  "progressive": "orthodox",
    "traditional": "modern",    "modern": "traditional",
    "conservative": "liberal",  "liberal": "conservative",

    # Employment / opportunity
    "employed": "unemployed",   "unemployed": "employed",
    "promoted": "demoted",      "hired": "fired",
    "included": "excluded",     "excluded": "included",
    "accepted": "rejected",     "rejected": "accepted",
    "welcomed": "shunned",      "shunned": "welcomed",
    "supported": "neglected",   "neglected": "supported",

    # Compound framing terms
    "discriminated": "privileged",
    "oppressed": "empowered",   "empowered": "oppressed",
    "suppressed": "liberated",  "liberated": "suppressed",
}

# ---------------------------------------------------------------------------
# Protected terms — NEVER swapped (prevent identity leakage)
# This set is intentionally broad; any identity marker must be preserved.
# ---------------------------------------------------------------------------
_IDENTITY_PROTECTED: FrozenSet[str] = frozenset({
    # Mask token
    "[mask]",
    # Religions
    "hindu", "hinduism", "hindus",
    "muslim", "islam", "muslims", "islamic",
    "sikh", "sikhism", "sikhs",
    "buddhist", "buddhism", "buddhists",
    "jain", "jainism", "jains",
    "christian", "christianity", "christians",
    # Broad caste categories
    "dalit", "dalits", "brahmin", "brahmins", "kshatriya", "kshatriyas",
    "vaishya", "shudra", "obc", "sc", "st",
    # Common caste names (subset; full list in evaluation/iir.py)
    "yadav", "jat", "patel", "reddy", "nair", "iyer", "iyengar",
    "chamar", "mahar", "mala", "valmiki", "paswan",
    "ansari", "qureshi", "sheikh", "syed", "pathan",
    "ahluwalia", "arora", "kamboj", "ramdasia",
    "agarwal", "oswal", "khandelwal",
    "chakma", "sherpa", "santhal",
    # Generic identity words
    "community", "caste", "religion", "religious", "faith",
    "minority", "minorities", "majority",
})


class Contrarium:
    """
    Minimal-perturbation counterfactual prompt builder (Appendix G).

    Perturbs contextually salient, **non-identity** tokens using a curated
    swap dictionary, leaving identity terms unchanged.

    Parameters
    ----------
    swap_dict : dict, optional
        Custom swap dictionary. Merged with (and overrides) the default pairs.
    extra_protected : set, optional
        Additional terms to protect from swapping (added to _IDENTITY_PROTECTED).

    Examples
    --------
    >>> c = Contrarium()
    >>> c.perturb("As a [MASK] applying for a job, I feel discriminated.")
    'As a [MASK] applying for a job, I feel privileged.'

    >>> c = Contrarium(swap_dict={"terrorist": "scholar"})
    >>> c.perturb("The [MASK] terrorist was arrested.")
    'The [MASK] scholar was arrested.'
    """

    def __init__(
        self,
        swap_dict: Optional[Dict[str, str]] = None,
        extra_protected: Optional[set] = None,
    ):
        self._swaps: Dict[str, str] = dict(CONTRARIUM_DEFAULT_SWAPS)
        if swap_dict:
            self._swaps.update(swap_dict)

        self._protected: FrozenSet[str] = _IDENTITY_PROTECTED
        if extra_protected:
            self._protected = self._protected | frozenset(
                t.lower() for t in extra_protected
            )

    def perturb(self, text: str) -> str:
        """
        Build a counterfactual by swapping eligible tokens in *text*.

        Preserves:
        - Original capitalisation of each token.
        - Punctuation attached to tokens.
        - All identity-protected terms (unchanged).
        - [MASK] token (always preserved; critical for IIR probing).

        Parameters
        ----------
        text : str
            Original prompt.

        Returns
        -------
        str
            Counterfactual prompt x'.
        """
        # Tokenise preserving punctuation with capturing groups
        pattern = re.compile(r"(\[MASK\]|\w[\w'-]*|\S)", re.IGNORECASE)
        result_parts = []
        last_end = 0

        for match in pattern.finditer(text):
            # Preserve whitespace between tokens
            result_parts.append(text[last_end:match.start()])
            token = match.group(0)
            last_end = match.end()

            # Never swap [MASK]
            if token.upper() == "[MASK]":
                result_parts.append(token)
                continue

            # Strip trailing punctuation for lookup
            stripped, suffix = self._split_punct(token)
            lower = stripped.lower()

            if lower in self._protected:
                result_parts.append(token)
            elif lower in self._swaps:
                replacement = self._swaps[lower]
                result_parts.append(self._match_case(replacement, stripped) + suffix)
            else:
                result_parts.append(token)

        result_parts.append(text[last_end:])
        return "".join(result_parts)

    def add_swap_pair(self, word: str, replacement: str, bidirectional: bool = True):
        """
        Dynamically add a swap pair.

        Parameters
        ----------
        word : str
        replacement : str
        bidirectional : bool
            If True, also add replacement → word.
        """
        self._swaps[word.lower()] = replacement.lower()
        if bidirectional:
            self._swaps[replacement.lower()] = word.lower()

    def get_swap_dict(self) -> Dict[str, str]:
        """Return a copy of the current swap dictionary."""
        return dict(self._swaps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_punct(token: str):
        """Separate trailing punctuation from a token word."""
        m = re.match(r"^([\w\[\]-]+)([\W]*)$", token)
        if m:
            return m.group(1), m.group(2)
        return token, ""

    @staticmethod
    def _match_case(replacement: str, original: str) -> str:
        """Preserve capitalisation style of original in replacement."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.title()
        return replacement
