"""Numerical contracts for factorization and inverse execution."""

import numpy as np
import pytest
import scipy.sparse as sp

from sansa import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig
from sansa.core.factorizers import GramianFactorizer


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
