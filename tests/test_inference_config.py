import pytest

from test import _resolve, resolve_prediction_threshold


def test_checkpoint_dtm_availability_is_default():
    assert _resolve("dtm_availability", None, 1.0, 0.0, False) == 1.0


def test_preprocessing_conflict_requires_explicit_override():
    with pytest.raises(ValueError, match="allow_preprocessing_override"):
        _resolve("dtm_availability", 0.0, 1.0, 1.0, False)
    assert _resolve("dtm_availability", 0.0, 1.0, 1.0, True) == 0.0


def test_zero_selected_threshold_is_not_replaced_by_fallback():
    assert resolve_prediction_threshold(None, 0.0) == 0.0
    assert resolve_prediction_threshold(None, None) == 0.5
    assert resolve_prediction_threshold(0.25, 0.75) == 0.25
