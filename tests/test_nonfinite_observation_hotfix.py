import math

import numpy as np

from pokebench30.representation.encoder import _accuracy_fraction, _clip


def test_clip_replaces_nan_with_zero():
    assert _clip(float("nan"), 0.0, 1.0) == 0.0


def test_clip_replaces_positive_infinity_with_zero():
    assert _clip(float("inf"), 0.0, 1.0) == 0.0


def test_clip_replaces_negative_infinity_with_zero():
    assert _clip(float("-inf"), -1.0, 1.0) == 0.0


def test_clip_preserves_finite_semantics():
    assert _clip(1.5, 0.0, 1.0) == 1.0
    assert _clip(-2.0, -1.0, 1.0) == -1.0
    assert _clip(0.25, 0.0, 1.0) == 0.25


def test_accuracy_nan_uses_unknown_accuracy_fallback():
    assert _accuracy_fraction(float("nan")) == 1.0


def test_accuracy_infinity_uses_unknown_accuracy_fallback():
    assert _accuracy_fraction(float("inf")) == 1.0
