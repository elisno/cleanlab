"""An internal wrapper around the Imagelab class from the CleanVision package to incorporate it into Datalab.
This allows low-quality images to be detected alongside other issues in computer vision datasets.
The methods/classes in this module are just intended for internal use.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.adapter.constants import (
    DEFAULT_CLEANVISION_ISSUES,
    IMAGELAB_ISSUES_MAX_PREVALENCE,
)
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.data_issues import DataIssues, _InfoStrategy
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter
from cleanlab.datalab.internal.task import Task

if TYPE_CHECKING:  # pragma: no cover
    from cleanvision import Imagelab
    from datasets.arrow_dataset import Dataset

from cleanvision.utils.viz_manager import VizManager


def create_imagelab(dataset: "Dataset", image_key: Optional[str]) -> Optional["Imagelab"]:
    """Creates Imagelab instance for running CleanVision checks. CleanVision checks are only supported for
    huggingface datasets as of now.

    Parameters
    ----------
    dataset: datasets.Dataset
        Huggingface dataset used by Imagelab
    image_key: str
        key for image feature in the huggingface dataset

    Returns
    -------
    Imagelab
    """
    imagelab = None
    if not image_key:
        return imagelab
    try:
        from cleanvision import Imagelab
        from datasets.arrow_dataset import Dataset

        if isinstance(dataset, Dataset):
            imagelab = Imagelab(hf_dataset=dataset, image_key=image_key)
        else:
            raise ValueError(
                "For now, only huggingface datasets are supported for running cleanvision checks inside cleanlab. You can easily convert most datasets to the huggingface dataset format."
            )

    except ImportError:
        raise ImportError(
            "Cannot import required image packages. Please install them via: `pip install cleanlab[image]` or just install cleanlab with "
            "all optional dependencies via: `pip install cleanlab[all]`"
        )
    return imagelab


class ImagelabDataIssuesAdapter(DataIssues):
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.
    strategy :
        Strategy used for processing info dictionaries.

    Parameters
    ----------
    issues : pd.DataFrame
        Stores information about each individual issue found in the data,
        on a per-example basis.
    issue_summary : pd.DataFrame
        Summarizes the overall statistics for each issue type.
    info : dict
        A dictionary that contains information and statistics about the data and each issue type.
    """

    def __init__(self, data: Data, strategy: Type[_InfoStrategy]) -> None:
        super().__init__(data, strategy)

    def _update_issues_imagelab(self, imagelab: "Imagelab", overlapping_issues: List[str]) -> None:
        overwrite_columns = [f"is_{issue_type}_issue" for issue_type in overlapping_issues]
        overwrite_columns.extend([f"{issue_type}_score" for issue_type in overlapping_issues])

        if overwrite_columns:
            warnings.warn(
                f"Overwriting columns {overwrite_columns} in self.issues with "
                f"columns from imagelab."
            )
            self.issues.drop(columns=overwrite_columns, inplace=True)
        new_columnns = list(set(imagelab.issues.columns).difference(self.issues.columns))
        self.issues = self.issues.join(imagelab.issues[new_columnns], how="outer")

    def filter_based_on_max_prevalence(self, issue_summary: pd.DataFrame, max_num: int):
        removed_issues = issue_summary[issue_summary["num_images"] > max_num]["issue_type"].tolist()
        if len(removed_issues) > 0:
            print(
                f"Removing {', '.join(removed_issues)} from potential issues in the dataset as it exceeds max_prevalence={IMAGELAB_ISSUES_MAX_PREVALENCE}"
            )
        return issue_summary[issue_summary["num_images"] <= max_num].copy()

    def collect_issues_from_imagelab(self, imagelab: "Imagelab", issue_types: List[str]) -> None:
        """
        Collect results from Imagelab and update datalab.issues and datalab.issue_summary

        Parameters
        ----------
        imagelab: Imagelab
            Imagelab instance that run all the checks for image issue types
        """
        overlapping_issues = list(set(self.issue_summary["issue_type"]) & set(issue_types))
        self._update_issues_imagelab(imagelab, overlapping_issues)

        if overlapping_issues:
            warnings.warn(
                f"Overwriting {overlapping_issues} rows in self.issue_summary from imagelab."
            )
        self.issue_summary = self.issue_summary[
            ~self.issue_summary["issue_type"].isin(overlapping_issues)
        ]
        imagelab_summary_copy = imagelab.issue_summary.copy()
        imagelab_summary_copy = self.filter_based_on_max_prevalence(
            imagelab_summary_copy, int(IMAGELAB_ISSUES_MAX_PREVALENCE * len(self.issues))
        )

        imagelab_summary_copy.rename({"num_images": "num_issues"}, axis=1, inplace=True)
        self.issue_summary = pd.concat(
            [self.issue_summary, imagelab_summary_copy], axis=0, ignore_index=True
        )
        for issue_type in issue_types:
            self._update_issue_info(issue_type, imagelab.info[issue_type])


class ImagelabReporterAdapter(Reporter):
    def __init__(
        self,
        data_issues: "DataIssues",
        imagelab: "Imagelab",
        task: Task,
        correlations_df: pd.DataFrame,
        verbosity: int = 1,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
    ):
        super().__init__(
            data_issues=data_issues,
            task=task,
            verbosity=verbosity,
            include_description=include_description,
            show_summary_score=show_summary_score,
            show_all_issues=show_all_issues,
        )
        self.imagelab = imagelab
        self.correlations_df = correlations_df
        self.threshold = 0.01

    def report(self, num_examples: int) -> None:
        super().report(num_examples)
        print("\n\n")
        self.imagelab.report(
            num_images=num_examples,
            max_prevalence=IMAGELAB_ISSUES_MAX_PREVALENCE,
            print_summary=False,
            verbosity=0,
            show_id=True,
        )

        correlated_properties = self._get_correlated_properties()
        if correlated_properties:
            print("\n\n")
            report_correlation_header = "Here is a summary of spurious correlations between image features like 'dark_score', 'blurry_score', etc., and class labels detected in the data.\n\n"
            report_correlation_metric = "A lower score for each property implies a higher correlation of that property with the class labels.\n\n"
            print(report_correlation_header + report_correlation_metric)
            filtered_correlations_df = self._get_filtered_correlated_properties(
                correlated_properties
            )
            print(filtered_correlations_df.to_string(index=False) + "\n")

            report_extremal_images = "Here are the images corresponding to the extreme (minimum and maximum) individual scores for each of the detected correlated properties:\n\n"
            print(report_extremal_images)
            issues = self.data_issues.get_issues()
            correlated_indices = {
                prop: [issues[prop].idxmin(), issues[prop].idxmax()]
                for prop in correlated_properties
            }
            self._visualize(correlated_indices, issues)

    def _visualize(self, correlated_indices, issues) -> None:
        for prop, image_ids in correlated_indices.items():
            print(
                f"{'Images with minimum and maximum individual scores for ' + prop.replace('_score', '') + ' issue:'}\n"
            )
            title_info = {"scores": [f"score: {issues.loc[id, prop]:.4f}" for id in image_ids]}
            VizManager.individual_images(
                images=[self.imagelab._dataset[id] for id in image_ids],
                title_info=title_info,
                ncols=2,
                cell_size=(2, 2),
            )

    def _get_correlated_properties(self) -> List:
        if self.correlations_df.empty:
            return []
        return self.correlations_df.query("score < @self.threshold")["property"].tolist()

    def _get_filtered_correlated_properties(self, correlated_properties: List) -> pd.DataFrame:
        filtered_correlations_df = self.correlations_df.query("property in @correlated_properties")
        filtered_correlations_df.loc[:, "property"] = filtered_correlations_df["property"].apply(
            lambda x: x.replace("_score", "")
        )
        return filtered_correlations_df


class ImagelabIssueFinderAdapter(IssueFinder):
    def __init__(self, datalab, task, verbosity):
        super().__init__(datalab, task, verbosity)
        self.imagelab = self.datalab._imagelab

    def _get_imagelab_issue_types(self, issue_types, **kwargs):
        if issue_types is None:
            return DEFAULT_CLEANVISION_ISSUES

        if "image_issue_types" not in issue_types:
            return None

        issue_types_copy = {}
        for issue_type, params in issue_types["image_issue_types"].items():
            if not params:
                issue_types_copy[issue_type] = DEFAULT_CLEANVISION_ISSUES[issue_type]
            else:
                issue_types_copy[issue_type] = params

        return issue_types_copy

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        datalab_issue_types = (
            {k: v for k, v in issue_types.items() if k != "image_issue_types"}
            if issue_types
            else issue_types
        )
        super().find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=datalab_issue_types,
        )

        issue_types_copy = self._get_imagelab_issue_types(issue_types)
        if not issue_types_copy:
            return
        try:
            if self.verbosity:
                print(f'Finding {", ".join(issue_types_copy.keys())} images ...')

            self.imagelab.find_issues(issue_types=issue_types_copy, verbose=False)

            self.datalab.data_issues.collect_statistics(self.imagelab)
            self.datalab.data_issues.collect_issues_from_imagelab(
                self.imagelab, issue_types_copy.keys()
            )
            if self.datalab.has_labels:
                self.datalab._correlations_df = self.datalab._spurious_correlation()
        except Exception as e:
            print(f"Error in checking for image issues: {e}")
