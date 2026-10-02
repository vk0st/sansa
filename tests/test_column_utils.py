"""Shared compressed-axis kernels and their model-level numerical contract."""

import numpy as np
import pytest
import scipy.sparse as sp

from sansa.utils import get_squared_norms_along_compressed_axis, inplace_scale_along_compressed_axis


def reference_norms(matrix):
    squared = np.zeros(matrix.nnz + 1)
    squared[:-1] = matrix.data**2
    return np.add.reduceat(squared, matrix.indptr[:-1]) * (np.diff(matrix.indptr) > 0)


def reference_scale(matrix, scale):
    matrix.data *= np.repeat(scale, np.diff(matrix.indptr))


def fixture_matrix(format, dtype, index_dtype):
    rng = np.random.default_rng(77)
    values = rng.normal(size=(9, 13)) * np.logspace(-8, 8, 13)
    values[rng.random(values.shape) < 0.5] = 0
    values[[0, 4, 8], :] = 0
    values[:, [0, 5, 12]] = 0
    matrix = getattr(sp, format + "_matrix")(values.astype(dtype))
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    return matrix


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_compressed_kernels_match_numpy(format, dtype, index_dtype):
    matrix = fixture_matrix(format, dtype, index_dtype)
    np.testing.assert_allclose(get_squared_norms_along_compressed_axis(matrix), reference_norms(matrix), rtol=1e-14, atol=0)
    assert get_squared_norms_along_compressed_axis(matrix).dtype == np.float64
    for scale_dtype in (np.float32, np.float64):
        scale = np.linspace(-2, 3, len(matrix.indptr) - 1, dtype=scale_dtype)
        expected, actual = matrix.copy(), matrix.copy()
        reference_scale(expected, scale)
        inplace_scale_along_compressed_axis(actual, scale)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.indices, matrix.indices)
        np.testing.assert_array_equal(actual.indptr, matrix.indptr)
    empty = getattr(sp, format + "_matrix")((9, 13), dtype=dtype)
    np.testing.assert_array_equal(get_squared_norms_along_compressed_axis(empty), np.zeros(len(empty.indptr) - 1))
    inplace_scale_along_compressed_axis(empty, np.ones(len(empty.indptr) - 1))


def test_scaling_validates_shape_and_preserves_numpy_casting():
    matrix = sp.eye(3, format="csc", dtype=np.float32)
    for scale in (np.ones(2), np.ones((3, 1))):
        with pytest.raises(ValueError, match="compressed segment"):
            inplace_scale_along_compressed_axis(matrix, scale)
        np.testing.assert_array_equal(matrix.data, np.ones(3))
    inplace_scale_along_compressed_axis(matrix, [1, 2, 3])
    np.testing.assert_array_equal(matrix.data, [1, 2, 3])
    # A view of the values is a valid scale vector; updates must not change it mid-loop.
    aliased = sp.csc_matrix(([2.0, 3.0, 4.0], [0, 0, 1], [0, 2, 3, 3]), shape=(3, 3))
    expected = aliased.copy()
    reference_scale(expected, expected.data)
    inplace_scale_along_compressed_axis(aliased, aliased.data)
    np.testing.assert_array_equal(aliased.data, expected.data)
    integer = sp.eye(3, format="csc", dtype=np.int32)
    with pytest.raises(TypeError):
        inplace_scale_along_compressed_axis(integer, np.full(3, 0.5))
    np.testing.assert_array_equal(integer.data, np.ones(3))
    # Integer squares deliberately retain NumPy's input-dtype overflow behavior.
    integer.data[:] = 50000
    np.testing.assert_array_equal(get_squared_norms_along_compressed_axis(integer), reference_norms(integer))


