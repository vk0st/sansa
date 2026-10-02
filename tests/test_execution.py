"""Numerical contracts for factorization and inverse execution."""

import numpy as np
import pytest
import scipy.sparse as sp

from sansa import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig
from sansa.core.factorizers import GramianFactorizer


def test_panel_prescaling_matches_complete_gram():
    from sansa.model import _apply_icf_scaling
    from sansa.utils import get_squared_norms_along_compressed_axis, inplace_scale_along_uncompressed_axis

    rng = np.random.default_rng(77)
    x = sp.random(16, 260, density=0.1, format="csr", dtype=np.float32, random_state=rng)
    expected = x.copy()
    norms = get_squared_norms_along_compressed_axis((x.T @ x).tocsc())
    scale = np.sqrt(np.sqrt(norms))
    scale[scale == 0] = 1
    inplace_scale_along_uncompressed_axis(expected, 1 / scale)
    _apply_icf_scaling(x, True)
    np.testing.assert_array_equal(x.data, expected.data)


def test_icf_keeps_discarded_candidate_diagonal_updates():
    from sansa.core._ops._factor_ops import icf

    # Row 1 is dropped from column 0, but its square must still reduce the next pivot.
    matrix = sp.csc_matrix([[9, 1, 2, 3], [1, 9, 0, 0], [2, 0, 9, 0], [3, 0, 0, 9]], dtype=np.float32)
    with pytest.raises(ValueError, match="2\\*n"):
        icf(sp.csc_matrix([[4, 1], [1, 4]], dtype=np.float32), l2=0.0, max_nnz=2)
    expected = np.sqrt(np.float32(9) - (np.float32(1) / np.float32(3)) ** 2)
    for dtype in (np.int32, np.int64):
        matrix.indices = matrix.indices.astype(dtype)
        matrix.indptr = matrix.indptr.astype(dtype)
        factor = icf(matrix, l2=0.0, max_nnz=8)
        np.testing.assert_array_equal(factor.indices[:factor.indptr[1]], [0, 2, 3])
        assert factor.diagonal()[1] == expected


@pytest.mark.parametrize("gramian", [False, True])
def test_modern_cholmod_owned_factor_and_long_indices(gramian):
    x = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.float32))
    system = (x.T @ x).tocsr()
    factorizer = GramianFactorizer.from_config(CHOLMODGramianFactorizerConfig(reordering_use_long=True))
    lower, permutation = factorizer.approximate_cholesky(x if gramian else system, 2.0, 1.0, gramian)
    np.testing.assert_allclose((lower @ lower.T).toarray(), (system.toarray() + 2 * np.eye(3))[np.ix_(permutation, permutation)], atol=1e-12)
    assert isinstance(lower, sp.csc_matrix) and lower.data.flags.writeable
    symbolic, _ = factorizer._analyze(x if gramian else system, gramian)
    assert symbolic.itype == np.dtype(np.int64)
    assert GramianFactorizer.from_config(ICFGramianFactorizerConfig()).factorization_method.value == "ICF"
