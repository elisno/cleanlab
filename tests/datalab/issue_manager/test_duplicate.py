from typing import Optional
import math

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra.numpy import arrays
from scipy.sparse import csr_matrix


from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager

SEED = 42


@composite
def embeddings_strategy(draw):
    shape_strategy = st.tuples(
        st.integers(min_value=3, max_value=20), st.integers(min_value=2, max_value=2)
    )
    element_strategy = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    embeddings = draw(
        arrays(
            dtype=np.float64,
            shape=shape_strategy,
            elements=element_strategy,
            unique=True,
        )
    )
    return embeddings

@composite
def knn_graph_strategy(draw, num_samples, k_neighbors):
    """This is a strategy used for creating a property based test for a `knn_graph: csr_matrix` object."""
    # Helper function to retrieve value from a strategy if given
    def get_value_or_draw(val):
        return draw(val) if isinstance(val, st.SearchStrategy) else val

    num_samples = get_value_or_draw(num_samples)
    k_neighbors = get_value_or_draw(k_neighbors)

    # Generate a symmetric distance matrix
    upper_triangle = [
        draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                     min_size=i, max_size=i))
        for i in range(1, num_samples + 1)
    ]

    distance_matrix = np.zeros((num_samples, num_samples))
    for i, row in enumerate(upper_triangle):
        distance_matrix[i, :i + 1] = row
        distance_matrix[:i + 1, i] = row

    np.fill_diagonal(distance_matrix, np.inf)  # Prevent selecting a point as its own neighbor

    # Compute k-nearest neighbors based on the distance matrix
    sorted_indices = np.argsort(distance_matrix, axis=1)
    kneighbor_indices = sorted_indices[:, :k_neighbors]
    kneighbor_distances = np.array([distance_matrix[i, kneighbor_indices[i]] for i in range(num_samples)])

    # Convert to CSR-representation
    data = kneighbor_distances.ravel()
    inds = kneighbor_indices.ravel()
    indptr = np.arange(0, num_samples * k_neighbors + 1, k_neighbors)

    return csr_matrix((data, inds, indptr), shape=(num_samples, num_samples))


def build_issue_manager(draw, num_samples_strategy, k_neighbors_strategy, with_issues: bool = False, threshold: Optional[float] = None):
    """This is a helper function to build a NearDuplicateIssueManager that finds near-duplicate issues for a given knn_graph.
    
    As it's used for property-based testing, the default thresholds have to be heavily increased to avoid flaky tests of small sample sizes.
    """
    knn_graph = draw(knn_graph_strategy(num_samples=num_samples_strategy, k_neighbors=k_neighbors_strategy))
    
    lab = Datalab(data={})
    _kwargs = {}
    if threshold is not None:
        _kwargs["threshold"] = threshold
    issue_manager = NearDuplicateIssueManager(datalab=lab, **_kwargs)
    issue_manager.find_issues(knn_graph=knn_graph)
    issues = issue_manager.issues["is_near_duplicate_issue"]
    
    if with_issues:
        assume(any(issues))
    else:
        assume(not any(issues))
    return issue_manager

@composite
def no_issue_issue_manager_strategy(draw):
    return build_issue_manager(draw, st.integers(min_value=10, max_value=50), st.integers(min_value=2, max_value=5), with_issues=False)

@composite
def issue_manager_with_issues_strategy(draw):
    return build_issue_manager(draw, st.integers(min_value=10, max_value=20), st.integers(min_value=2, max_value=5), with_issues=True, threshold=0.9)


class TestNearDuplicateIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = (
            embeddings_array[3, :] + np.random.rand(embeddings_array.shape[1]) * 0.001
        )
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab, embeddings, monkeypatch):
        mock_data = lab.data.from_dict({**lab.data.to_dict(), **embeddings})
        monkeypatch.setattr(lab, "data", mock_data)
        return NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,
        )

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab
        assert issue_manager.metric == "euclidean"
        assert issue_manager.k == 2
        assert issue_manager.threshold == 0.13

        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            threshold=0.1,
        )
        assert issue_manager.threshold == 0.1

    def test_find_issues(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 3 + [True] * 2)
        assert np.all(
            issues["is_near_duplicate_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "near_duplicate"
        assert summary["score"][0] == pytest.approx(expected=0.03122489, abs=1e-7)

        assert (
            info.get("near_duplicate_sets", None) is not None
        ), "Should have sets of near duplicates"

        new_issue_manager = NearDuplicateIssueManager(
            datalab=issue_manager.datalab,
            metric="euclidean",
            k=2,
            threshold=0.1,
        )
        new_issue_manager.find_issues(features=embeddings["embedding"])

    def test_report(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "------------------ near_duplicate issues -------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report

    @given(embeddings=embeddings_strategy())
    @settings(deadline=800)
    def test_near_duplicate_sets(self, embeddings):
        data = {"metadata": ["" for _ in range(len(embeddings))]}
        lab = Datalab(data)
        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,
        )
        embeddings = np.array(embeddings)
        issue_manager.find_issues(features=embeddings)
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        issues = issue_manager.issues["is_near_duplicate_issue"]

        # Test: Near duplicates are symmetric
        all_symmetric = all(
            i in near_duplicate_sets[j]
            for i, near_duplicates in enumerate(near_duplicate_sets)
            for j in near_duplicates
        )
        assert all_symmetric, "Some near duplicate sets are not symmetric"

        # Test: Near duplicate sets for issues
        assert all(
            len(near_duplicate_set) == 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if not issues[i]
        ), "Non-issue examples should not have near duplicate sets"
        assert all(
            len(near_duplicate_set) > 0
            for i, near_duplicate_set in enumerate(near_duplicate_sets)
            if issues[i]
        ), "Issue examples should have near duplicate sets"

class TestKNNGraphRepresentation:
    """Validates the properties of the knn_graph objects as they are computed by scikit-learn's NearestNeighbors.kneighbors_graph method.
    
    This is a helper test class to make sure we're testing knn-graph-based issue checkers with valid knn-graphs.
    """
    # Test the strategy to check if it works as expected
    @given(knn_graph=knn_graph_strategy(num_samples=st.integers(min_value=10, max_value=50), k_neighbors=st.integers(min_value=2, max_value=5)))
    def test_knn_graph(self, knn_graph):
        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        indices = knn_graph.indices.reshape(N, -1)
    
        # Validation checks
        self._check_distances_sorted(distances)
        self._check_indices_validity(indices)
        self._verify_mutual_neighbors_have_same_distances(distances, indices, N)
        self._verify_mutual_consistency_of_distances(distances, indices, N)

    def _check_distances_sorted(self, distances) -> None:
        # Check distances are sorted in ascending order for each row
        for row in distances:
            assert all(row[i] <= row[i+1] for i in range(len(row)-1))

    def _check_indices_validity(self, indices) -> None:
        # Check that indices are unique across columns and don't have the row's index
        for row_idx, row in enumerate(indices):
            assert len(set(row)) == len(row)
            assert row_idx not in row

    def _verify_mutual_neighbors_have_same_distances(self, distances, indices, N) -> None:
        # Verify that mutual neighbors have the same distances
        for i in range(N):
            for j in indices[i]:
                if i in indices[j]:
                    d_ij = distances[i][list(indices[i]).index(j)]
                    d_ji = distances[j][list(indices[j]).index(i)]
                    assert math.isclose(d_ij, d_ji), f"Distances between {i} and {j} do not match: {d_ij} vs {d_ji}"
    
    def _verify_mutual_consistency_of_distances(self, distances, indices, N) -> None:
        # Verify the mutual consistency of k-NN distances:
        # For every point i and its neighbor j, ensure that the distance from i to j 
        # cannot be smaller than the distance from any other neighbor k of j to j. 
        for i in range(N):
            for j in indices[i]:
                d_ij = distances[i][list(indices[i]).index(j)]
                j_neighbors_distances = distances[j]
                if d_ij < max(j_neighbors_distances):
                    assert i in indices[j], f"Point {i} should be a neighbor of point {j}, it's closer than the farthest neighbor of {j}"


class TestNearDuplicateSets:
    
    @given(issue_manager=no_issue_issue_manager_strategy())
    @settings(deadline=800)
    def test_near_duplicate_sets_empty_if_no_issue(self, issue_manager):
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        assert all(len(near_duplicate_set) == 0 for near_duplicate_set in near_duplicate_sets)
        
    @given(issue_manager=issue_manager_with_issues_strategy())
    @settings(deadline=800, max_examples=1000)
    def test_symmetric_and_flagged_consistency(self, issue_manager):
        near_duplicate_sets = issue_manager.info["near_duplicate_sets"]
        issues = issue_manager.issues["is_near_duplicate_issue"]
        
        # Test symmetry: If A is in near_duplicate_set of B, then B should be in near_duplicate_set of A.
        for i, near_duplicates in enumerate(near_duplicate_sets):
            for j in near_duplicates:
                assert i in near_duplicate_sets[j], f"Example {j} is in near_duplicate_set of {i}, but not vice versa"

                
        # If an example is flagged as near_duplicate, then every example in its near_duplicate_set should also be flagged as near_duplicate.
        for i, near_duplicate_set in enumerate(near_duplicate_sets):
            if issues[i]:
                # Near duplicate sets of flagged examples should not be empty
                assert len(near_duplicate_set) > 0, "Near duplicate set of flagged example should not be empty"
                
                # Examples in near_duplicate_set should also be flagged as near_duplicate
                set_flags = issues[np.array(list(near_duplicate_set))]
                assert all(set_flags), f"Example {i} is flagged as near_duplicate but some examples in its near_duplicate_set are not"
            else:
                # Near duplicate sets of non-flagged examples should be empty
                assert len(near_duplicate_set) == 0, "Near duplicate set of non-flagged example should be empty"
