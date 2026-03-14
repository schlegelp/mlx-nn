import numpy as np
import pytest

from mlx_nn.core import KNN


@pytest.mark.parametrize("k", [1, 3])
def test_knn_matches_scipy_sqeuclidean(k):
    """Verify mlx_nn.KNN returns the same neighbors as scipy.spatial.distance."""

    # Use a fixed seed for deterministic results
    rng = np.random.default_rng(0)

    # Mid-sized dataset: 1000 points in 10 dimensions, 3 query points
    data = rng.standard_normal(size=(1000, 10)).astype(np.float32)
    queries = rng.standard_normal(size=(3, 10)).astype(np.float32)

    knn = KNN(data)
    dists, idx = knn.query(queries, k=k, return_numpy=True)

    # Compute ground-truth with scipy
    from scipy.spatial.distance import cdist

    # KNN.query returns squared Euclidean distances
    sqdists = cdist(queries, data, metric="sqeuclidean")
    expected_idx = np.argsort(sqdists, axis=1)[:, :k]
    expected_dists = np.take_along_axis(sqdists, expected_idx, axis=1)

    assert idx.shape == expected_idx.shape
    assert dists.shape == expected_dists.shape

    # Compare indices & distances
    np.testing.assert_array_equal(idx, expected_idx)
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-6, atol=1e-6)
