from typing import cast
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from cleanlab.internal.neighbor import features_to_knn
from cleanlab.internal.neighbor.neighbor import (
    correct_knn_distances_and_indices,
    correct_knn_graph,
    knn_to_knn_graph,
)


@pytest.mark.parametrize(
    "N",
    [2, 10, 100, 101],
    ids=lambda x: f"N={x}",
)
@pytest.mark.parametrize(
    "M",
    [2, 3, 4, 5, 10, 50, 100],
    ids=lambda x: f"M={x}",
)
def test_features_to_knn(N, M):

    features = np.random.rand(N, M)
    knn = features_to_knn(features)

    assert isinstance(knn, NearestNeighbors)
    knn = cast(NearestNeighbors, knn)
    assert knn.n_neighbors == min(10, N - 1)
    if M > 3:
        metric = knn.metric
        assert metric == "cosine"
    else:
        metric = knn.metric
        if N <= 100:
            assert hasattr(metric, "__name__")
            metric = metric.__name__
        assert metric == "euclidean"

    # The knn object should be fitted to the features.
    # TODO: This is not a good test, but it's the best we can do without exposing the internal state of the knn object.
    # Assert is_
    assert knn._fit_X is features


def test_knn_kwargs():
    """Check that features_to_knn passes additional keyword arguments to the NearestNeighbors constructor correctly."""
    features = np.random.rand(100, 10)
    V = features.var(axis=0)
    knn = features_to_knn(features, n_neighbors=6, metric="seuclidean", metric_params={"V": V})

    assert knn.n_neighbors == 6
    assert knn.metric == "seuclidean"
    assert knn._fit_X is features
    assert knn.metric_params == {"V": V}


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_knn_to_knn_graph(metric):
    N, k = 100, 10
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    X = np.random.rand(N, 10)
    knn.fit(X)
    knn_graph = knn_to_knn_graph(knn)

    assert knn_graph.shape == (N, N)
    assert knn_graph.nnz == N * k
    assert knn_graph.dtype == np.float64
    assert np.all(knn_graph.data >= 0)
    assert np.all(knn_graph.indices >= 0)
    assert np.all(knn_graph.indices < 100)

    distances = knn_graph.data.reshape(N, k)
    indices = knn_graph.indices.reshape(N, k)

    # Assert all rows in distances are sorted
    assert np.all(np.diff(distances, axis=1) >= 0)


class TestKNNCorrection:
    def test_knn_graph_corrects_missing_duplicates(self):
        """Test that the KNN graph correction identifies missing duplicates and places them correctly."""
        X = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        )

        # k = 2
        retrieved_distances = np.array(
            [
                [0, np.sqrt(2)],
                [0, 0],
                [0, 0],
                [np.sqrt(2)] * 2,
            ]
        )
        retrieved_indices = np.array(
            [
                [1, 3],
                [0, 2],
                [0, 1],
                [0, 1],
            ]
        )

        # Most of the retrieved distances are correct, except for the first row that has two exact duplicates
        expected_distances = np.copy(retrieved_distances)
        expected_distances[0] = [0, 0]
        expected_indices = np.copy(retrieved_indices)
        expected_indices[0] = [1, 2]

        # Simulate an properly ordered KNN graph, which missed an exact duplicate in row 0
        knn_graph = csr_matrix(
            (retrieved_distances.ravel(), retrieved_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )
        expected_knn_graph = csr_matrix(
            (expected_distances.ravel(), expected_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )

        # Test that the distances and indices are corrected
        corrected_distances, corrected_indices = correct_knn_distances_and_indices(
            X, retrieved_distances, retrieved_indices
        )
        np.testing.assert_array_equal(corrected_distances, expected_distances)
        np.testing.assert_array_equal(corrected_indices, expected_indices)

        # Test that the knn graph can be corrected as well
        corrected_knn_graph = correct_knn_graph(X, knn_graph)
        np.testing.assert_array_equal(corrected_knn_graph.toarray(), expected_knn_graph.toarray())

    def test_knn_graph_corrects_order_of_duplicates(self):
        """Ensure that KNN correction prioritizes duplicates correctly even when initial indices are out of order."""
        X = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        )

        retrieved_distances = np.array(
            [
                [np.sqrt(2), 0],  # Should be [0, 0]
                [0, 0],
                [0, 0],
                [np.sqrt(2)] * 2,
            ]
        )
        retrieved_indices = np.array(
            [
                [3, 1],  # Should be [1, 2]
                [0, 2],
                [1, 0],  # Should be [0, 1]
                [0, 1],
            ]
        )

        expected_distances = np.copy(retrieved_distances)
        expected_distances[0] = [0, 0]

        expected_indices = np.copy(retrieved_indices)
        expected_indices[0] = [1, 2]
        expected_indices[2] = [0, 1]

        # Simulate an IMPROPERLY ordered KNN graph
        knn_graph = csr_matrix(
            (retrieved_distances.ravel(), retrieved_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )
        expected_knn_graph = csr_matrix(
            (expected_distances.ravel(), expected_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )

        # Test that the distances and indices are corrected
        corrected_distances, corrected_indices = correct_knn_distances_and_indices(
            X, retrieved_distances, retrieved_indices
        )
        np.testing.assert_array_equal(corrected_distances, expected_distances)
        np.testing.assert_array_equal(corrected_indices, expected_indices)

        # Test that the knn graph can be corrected as well
        corrected_knn_graph = correct_knn_graph(X, knn_graph)
        np.testing.assert_array_equal(corrected_knn_graph.toarray(), expected_knn_graph.toarray())
