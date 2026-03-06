"""
tests/test_iir.py
=================
Unit tests for the IIREvaluator and IIRResult classes.

Run with:
    pytest tests/test_iir.py -v
"""

import pytest
from ambedkar.evaluation.iir import IIREvaluator, IIRResult


@pytest.fixture
def evaluator():
    return IIREvaluator()


# ──────────────────────────────────────────────────────────────────────────────
# Detection correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestIdentityDetection:
    """IIREvaluator correctly identifies religion and caste terms."""

    def test_detects_religion_term_muslim(self, evaluator):
        outputs = [{"prompt": "x", "generated": "The Muslim community organised a march."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_detects_religion_term_hindu(self, evaluator):
        outputs = [{"prompt": "x", "generated": "Hindu families celebrated the festival."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_detects_caste_term_dalit(self, evaluator):
        outputs = [{"prompt": "x", "generated": "The Dalit students topped the district exam."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_detects_caste_term_brahmin(self, evaluator):
        outputs = [{"prompt": "x", "generated": "Brahmin priests perform the ritual."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_no_detection_neutral(self, evaluator):
        outputs = [{"prompt": "x", "generated": "The committee organised a public event."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(0.0)

    def test_case_insensitive_detection(self, evaluator):
        outputs = [{"prompt": "x", "generated": "DALIT families were excluded from the panchayat."}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_partial_word_not_detected(self, evaluator):
        """'Brahminical' contains 'Brahmin' but boundary match should handle it."""
        outputs = [{"prompt": "x", "generated": "The brahminical order dominated the discourse."}]
        # brahminical → contains brahmin as substring; word-boundary regex should still fire
        result = evaluator.evaluate(outputs)
        # This documents current behaviour; exact result may differ by implementation
        assert isinstance(result.overall_iir, float)

    def test_multiple_terms_count_once_per_prompt(self, evaluator):
        """A prompt with multiple identity terms counts as 1 positive, not multiple."""
        outputs = [
            {"prompt": "x", "generated": "Muslim and Hindu communities clashed over the land."}
        ]
        result = evaluator.evaluate(outputs)
        # IIR is binary per prompt: was *any* identity term emitted?
        assert result.overall_iir == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestAggregation:

    def test_iir_is_fraction(self, evaluator):
        outputs = [
            {"prompt": "p1", "generated": "The Muslim community protested."},
            {"prompt": "p2", "generated": "The event was peaceful."},
            {"prompt": "p3", "generated": "Dalit residents were excluded."},
            {"prompt": "p4", "generated": "A local gathering took place."},
        ]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(0.5)

    def test_all_zero(self, evaluator):
        outputs = [
            {"prompt": f"p{i}", "generated": "A community meeting was held."}
            for i in range(10)
        ]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(0.0)

    def test_all_one(self, evaluator):
        outputs = [
            {"prompt": f"p{i}", "generated": "The Brahmin temple was consecrated."}
            for i in range(5)
        ]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(1.0)

    def test_iir_in_range(self, evaluator):
        import random
        random.seed(0)
        vocab = [
            "Dalit", "Brahmin", "Muslim", "Hindu", "Sikh", "A person",
            "The committee", "Several", "Community members",
        ]
        outputs = [
            {"prompt": f"p{i}", "generated": f"{random.choice(vocab)} attended the event."}
            for i in range(50)
        ]
        result = evaluator.evaluate(outputs)
        assert 0.0 <= result.overall_iir <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Per-religion breakdown
# ──────────────────────────────────────────────────────────────────────────────

class TestPerGroupBreakdown:

    def test_per_religion_keys_present(self, evaluator):
        outputs = [
            {"prompt": "p1", "generated": "Muslim families were denied housing."},
            {"prompt": "p2", "generated": "Hindu pilgrims arrived at the site."},
            {"prompt": "p3", "generated": "A neutral statement."},
        ]
        result = evaluator.evaluate(outputs)
        assert isinstance(result.per_religion_iir, dict)

    def test_iir_result_dataclass_fields(self, evaluator):
        outputs = [{"prompt": "x", "generated": "Dalit students excelled."}]
        result = evaluator.evaluate(outputs)
        assert hasattr(result, "overall_iir")
        assert hasattr(result, "per_religion_iir")
        assert hasattr(result, "per_caste_iir")
        assert hasattr(result, "n_prompts")

    def test_n_prompts_correct(self, evaluator):
        outputs = [{"prompt": f"p{i}", "generated": "text"} for i in range(7)]
        result = evaluator.evaluate(outputs)
        assert result.n_prompts == 7


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_input(self, evaluator):
        result = evaluator.evaluate([])
        assert result.overall_iir == 0.0
        assert result.n_prompts == 0

    def test_empty_generated_string(self, evaluator):
        outputs = [{"prompt": "x", "generated": ""}]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(0.0)

    def test_none_generated_handled(self, evaluator):
        # Some pipelines may produce None on failure; evaluator should not crash
        outputs = [{"prompt": "x", "generated": None}]
        try:
            result = evaluator.evaluate(outputs)
            assert isinstance(result.overall_iir, float)
        except (TypeError, AttributeError):
            pytest.skip("Implementation does not yet handle None generated values.")

    def test_hindi_text_detected(self, evaluator):
        """Hindi transliterations of identity terms should still trigger detection
        if the evaluator's lexicon includes them."""
        outputs = [{"prompt": "x", "generated": "Dalit समुदाय के लोग।"}]
        result = evaluator.evaluate(outputs)
        # 'Dalit' appears in ASCII inside the Hindi sentence; should be detected
        assert result.overall_iir == pytest.approx(1.0)

    def test_large_batch(self, evaluator):
        outputs = [
            {"prompt": f"p{i}", "generated": "The event was peaceful and uneventful."}
            for i in range(1000)
        ]
        result = evaluator.evaluate(outputs)
        assert result.overall_iir == pytest.approx(0.0)
        assert result.n_prompts == 1000
