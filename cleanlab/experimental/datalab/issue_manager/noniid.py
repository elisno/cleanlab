from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, cast
import warnings

import scipy
import numpy as np
import pandas as pd
import numpy.typing as npt
from fast_histogram import histogram1d # TODO new dependency
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab import Datalab


#TODO typing and method signatures

class NonIIDIssueManager(IssueManager):  # pragma: no cover
    """Manages issues related to non-iid data distributions."""

    description: ClassVar[
        str
    ] = """ TODO add descriptions
    """
    issue_name: ClassVar[str] = "non_iid"
    verbosity_levels = {
        0: {"issue": ["p-value"]},
        1: {},
        2: {"issue": ["nearest_neighbor", "distance_to_nearest_neighbor"]},
        }

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = "cosine",
        threshold: Optional[float] = None,
        k: Optional[int] = 10,
        num_permutations: Optional[int] = 25,
        **_,
    ):

        super().__init__(datalab)
        self.metric = metric
        self.threshold = threshold
        self.k = k
        self.num_permutations = num_permutations
        self.knn: Optional[NearestNeighbors] = None
        self._embeddings: Optional[npt.NDArray] = None
        self.tests = {
            'ks': self._ks_test, # TODO rename test
        }
        # TODO

    def find_issues(
        self,
            features: npt.NDArray,
            **_,
    ) -> None:

        self._embeddings = features

        if self.knn is None:
            if self.metric is None:
                self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
            self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

        if self.metric and self.metric != self.knn.metric:
            warnings.warn(
                f"Metric {self.metric} does not match metric {self.knn.metric} used to fit knn. "
                "Most likely an existing NearestNeighbors object was passed in, but a different "
                "metric was specified."
            )
        self.metric = self.knn.metric

        try:
            check_is_fitted(self.knn)
        except:
            self.knn.fit(self._embeddings)

        self.neighbor_graph = self._get_neighbor_graph()

        self.num_neighbors = self.k
        self.num_non_neighbors = min(10 * self.num_neighbors, len(self.neighbor_graph) - self.num_neighbors - 1)
        self.neighbor_index_distances = self._sample_neighbors(num_samples=self.num_neighbors)
        self.non_neighbor_index_distances = self._sample_non_neighbors(num_samples=self.num_non_neighbors)
        neighbor_histogram = self._build_histogram(self.neighbor_index_distances.flatten())
        non_neighbor_histogram = self._build_histogram(self.non_neighbor_index_distances.flatten())

        self.statistics = self._get_statistics(
            neighbor_histogram,
            non_neighbor_histogram
        )

        self.p_value = self._permutation_test(num_permutations=self.num_permutations)

        # TODO what about scores?
        scores = self._score_dataset()
        self.issues = pd.DataFrame(
            {
                # f"is_{self.issue_name}_issue": scores < self.threshold,  # TODO this doesn't make sense
                self.issue_score_key: scores,
            },
        )

        self.summary = self.get_summary(score=self.p_value) # TODO is the p-value the right thing to include here?

        self.info = self.collect_info()

    def collect_info(self) -> dict:
        issues_dict = {
            "p-value": self.p_value,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }
        
        weighted_knn_graph = self.knn.kneighbors_graph(mode="distance")  # type: ignore[union-attr]

        knn_info_dict = {
            "weighted_knn_graph": weighted_knn_graph.toarray().tolist(),
        }


        info_dict = {
            **issues_dict,
            **params_dict,
            **knn_info_dict,
        }
        return info_dict

    def _permutation_test(self, num_permutations) -> float:
        graph = self.neighbor_graph
        tiled = np.tile(np.arange(len(graph)), (len(graph), 1))
        index_distances = tiled - tiled.transpose()
        neighbors = graph > 0
        others = graph < 0

        statistics = []
        for i in range(num_permutations):
            perm = np.random.permutation(len(graph))
            distance = (perm - np.arange(len(graph))).reshape(len(graph), 1)
            new_distances = np.abs(distance - index_distances - distance.transpose())
        
            neighbor_index_distances = self._sample_neighbors(distances=new_distances, num_samples=self.num_neighbors).flatten()
            non_neighbor_index_distances = self._sample_non_neighbors(distances=new_distances, num_samples=self.num_non_neighbors).flatten()
            neighbor_histogram = self._build_histogram(neighbor_index_distances)
            non_neighbor_histogram = self._build_histogram(non_neighbor_index_distances)
        
            stats = self._get_statistics(
                neighbor_histogram,
                non_neighbor_histogram,
            )
            statistics.append(stats)
            
        ks_stats = np.array([stats['ks'] for stats in statistics])
        ks_stats_kde = scipy.stats.gaussian_kde(ks_stats)
        p_value = ks_stats_kde.integrate_box(self.statistics['ks'], 100)

        return p_value

    def _score_dataset(self) -> dict[int, float]:
        graph = self.neighbor_graph
        scores = {}    
        
        num_bins = len(graph) - 1
        bin_range = (1, num_bins)
    
        neighbor_cdfs = self._compute_row_cdf(self.neighbor_index_distances, num_bins, bin_range)
        non_neighbor_cdfs = self._compute_row_cdf(self.non_neighbor_index_distances, num_bins, bin_range)

        stats = np.sum(np.abs(neighbor_cdfs - non_neighbor_cdfs), axis=1)

        indices = np.arange(len(graph))
        reverse = len(graph) - indices
        normalizer = np.where(indices > reverse, indices, reverse)

        scores = stats / normalizer
        scores = np.tanh(-1 * scores) + 1
        
        scores = {idx: scores[idx] for idx in range(len(scores))}
        return scores

    def _compute_row_cdf(self, array, num_bins, bin_range) -> np.ndarray:
        histograms = np.apply_along_axis(lambda x: histogram1d(x, num_bins, bin_range), 1, array)
        histograms = histograms / np.sum(histograms[0])

        cdf = np.apply_along_axis(np.cumsum, 1, histograms)
        return cdf

    def _get_neighbor_graph(self) -> np.ndarray:
        """
        Given a fitted knn object, returns an array in which A[i,j] = n if
        item i and j are nth nearest neighbors. For n > k, A[i,j] = -1. Additionally, A[i,i] = 0
        """
        
        distances, kneighbors = self.knn.kneighbors()
        graph = self.knn.kneighbors_graph(n_neighbors=self.k).toarray()

        kneighbor_graph = np.ones(graph.shape) * -1
        for i, nbrs in enumerate(kneighbors):
            kneighbor_graph[i,nbrs] = 1 + np.arange(len(nbrs))
            kneighbor_graph[i,i] = 0
        return kneighbor_graph

    def _ks_test( # TODO change name
            self,
            neighbor_histogram,
            non_neighbor_histogram,
    ) -> float:
        neighbor_cdf = np.array([np.sum(neighbor_histogram[:i]) for i in range(len(neighbor_histogram) + 1)])
        non_neighbor_cdf = np.array([np.sum(non_neighbor_histogram[:i]) for i in range(len(non_neighbor_histogram) + 1)])
    
        statistic = np.max(np.abs(neighbor_cdf - non_neighbor_cdf))
        return statistic

    def _get_statistics(
            self,
            neighbor_index_distances,
            non_neighbor_index_distances,
    ) -> dict[str, float]:

        statistics = {}
        for key, test in self.tests.items():
            statistic = test(
                neighbor_index_distances,
                non_neighbor_index_distances,
            )
            statistics[key] = statistic
        return statistics

    def _sample_distances(
            self,
            sample_neighbors,
            distances=None,
            num_samples=1) -> np.ndarray:
        graph = self.neighbor_graph
        all_idx = np.arange(len(graph))
        all_idx = np.tile(all_idx, (len(graph),1))
        if sample_neighbors:
            indices = all_idx[graph > 0].reshape(len(graph), -1)
        else:
            indices = all_idx[graph < 0].reshape(len(graph), -1)
        generator = np.random.default_rng()
        choices = generator.choice(indices, axis=1, size=num_samples, replace=False)
        if distances is None:
            sample_distances = np.abs(np.arange(len(graph)) - choices.transpose()).transpose()
        else:
            sample_distances = distances[np.arange(len(graph)), choices.transpose()].transpose()
        return sample_distances

    def _sample_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(sample_neighbors=True, distances=distances, num_samples=num_samples)

    def _sample_non_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(sample_neighbors=False, distances=distances, num_samples=num_samples)

    def _build_histogram(self, index_array) -> np.ndarray:
        num_bins = len(self.neighbor_graph) - 1
        bin_range = (1, num_bins)
        histogram = histogram1d(index_array, num_bins, bin_range)
        histogram = histogram /  len(index_array)
        return histogram

    

    