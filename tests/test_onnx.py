import numpy as np
import pytest

from docguard_ai.models.export_onnx import check_equivalence


def test_equivalence_passes_within_tolerance():
    a = np.array([0.1, 0.9], dtype=np.float32)
    check_equivalence(a, a + 1e-6, atol=1e-4)


def test_equivalence_fails_outside_tolerance():
    a = np.array([0.1, 0.9], dtype=np.float32)
    with pytest.raises(AssertionError):
        check_equivalence(a, a + 0.1, atol=1e-4)
