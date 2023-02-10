# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


import os
import pickle
from unittest.mock import Mock, patch
from cleanlab.experimental.datalab.datalab import Datalab

from sklearn.neighbors import NearestNeighbors
from datasets.dataset_dict import DatasetDict
import numpy as np
import pandas as pd

from pathlib import Path

import pytest


def test_datalab_invalid_datasetdict(dataset, label_name):
    with pytest.raises(ValueError) as e:
        datadict = DatasetDict({"train": dataset, "test": dataset})
        Datalab(datadict, label_name)  # type: ignore
        assert "Please pass a single dataset, not a DatasetDict." in str(e)


class TestDatalab:
    """Tests for the Datalab class."""

    @pytest.fixture
    def lab(self, dataset, label_name):
        return Datalab(data=dataset, label_name=label_name)

    def test_print(self, lab, capsys):
        # Can print the object
        print(lab)
        captured = capsys.readouterr()
        assert "Datalab\n" == captured.out

    def tmp_path(self):
        # A path for temporarily saving the instance during tests.
        # This is a workaround for the fact that the Datalab class
        # does not have a save method.
        return Path(__file__).parent / "tmp.pkl"

    def test_attributes(self, lab):
        # Has the right attributes
        for attr in ["data", "label_name", "_labels", "info", "issues"]:
            assert hasattr(lab, attr), f"Missing attribute {attr}"

        assert all(lab._labels == np.array([1, 1, 2, 0, 2]))
        assert isinstance(lab.issues, pd.DataFrame), "Issues should by in a dataframe"
        assert isinstance(lab.issue_summary, pd.DataFrame), "Issue summary should be a dataframe"

    def test_get_info(self, lab):
        """Test that the method fetches the values from the info dict."""
        data_info = lab.get_info("data")
        assert data_info["num_classes"] == 3
        lab.info["data"]["num_classes"] = 4
        data_info = lab.get_info("data")
        assert data_info["num_classes"] == 4

    @pytest.mark.parametrize(
        "issue_types",
        [None, {"label": {}}],
        ids=["Default issues", "Only label issues"],
    )
    def test_find_issues(self, lab, pred_probs, issue_types):
        assert lab.issues.empty, "Issues should be empty before calling find_issues"
        assert lab.issue_summary.empty, "Issue summary should be empty before calling find_issues"
        lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        assert not lab.issues.empty, "Issues weren't updated"
        assert not lab.issue_summary.empty, "Issue summary wasn't updated"

        if issue_types is None:
            # Test default issue types
            columns = lab.issues.columns
            for issue_type in ["label", "outlier"]:
                assert f"is_{issue_type}_issue" in columns
                assert f"{issue_type}_score" in columns

    def test_find_issues_with_custom_hyperparams(self, lab, pred_probs):
        dataset_size = lab.get_info("data")["num_examples"]
        embedding_size = 2
        mock_embeddings = np.random.rand(dataset_size, embedding_size)

        knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
        issue_types = {
            "outlier": {
                "ood_kwargs": {"params": {"knn": knn}},
            },
        }
        lab.find_issues(
            pred_probs=pred_probs,
            features=mock_embeddings,
            issue_types=issue_types,
        )
        metric = lab.info["outlier"]["metric"]
        assert metric == "euclidean"
        n_neighbors = lab.info["outlier"]["n_neighbors"]
        assert n_neighbors == 3

    def test_validate_issue_types_dict(self, lab, monkeypatch):
        issue_types = {
            "issue_type_1": {f"arg_{i}": f"value_{i}" for i in range(1, 3)},
            "issue_type_2": {f"arg_{i}": f"value_{i}" for i in range(1, 4)},
        }
        defaults_dict = issue_types.copy()

        issue_types["issue_type_2"][
            "arg_2"
        ] = "another_value_2"  # Should be in default, but not affect the test
        issue_types["issue_type_2"][
            "arg_4"
        ] = "value_4"  # Additional arg not in defaults should be allowed (ignored)

        with monkeypatch.context() as m:
            m.setitem(issue_types, "issue_type_1", {})
            with pytest.raises(ValueError) as e:
                lab._validate_issue_types_dict(issue_types, defaults_dict)
            assert all([string in str(e.value) for string in ["issue_type_1", "arg_1", "arg_2"]])

    @pytest.mark.parametrize(
        "defaults_dict",
        [
            {"issue_type_1": {"arg_1": "default_value_1"}},
        ],
    )
    @pytest.mark.parametrize(
        "issue_types",
        [{"issue_type_1": {"arg_1": "value_1", "arg_2": "value_2"}}, {"issue_type_1": {}}],
    )
    def test_set_issue_types(self, lab, issue_types, defaults_dict, monkeypatch):
        """Test that the issue_types dict is set correctly."""
        with monkeypatch.context() as m:
            # Mock the validation method to do nothing
            m.setattr(lab, "_validate_issue_types_dict", lambda x, y: None)
            issue_types_copy = lab._set_issue_types(issue_types, defaults_dict)

            # For each argument in issue_types missing from defaults_dict, it should be added to the defaults dict
            for issue_type, args in issue_types.items():
                missing_args = set(args.keys()) - set(defaults_dict[issue_type].keys())
                for arg in missing_args:
                    assert issue_types_copy[issue_type][arg] == args[arg]

    # Mock the lab.issues dataframe to have some pre-existing issues
    def test_update_issues(self, lab, pred_probs, monkeypatch):
        """If there are pre-existing issues in the lab,
        find_issues should add columns to the issues dataframe for each example.
        """
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        # Check that the issues dataframe has the right columns
        expected_issues_df = pd.DataFrame(
            {
                "is_foo_issue": mock_issues.is_foo_issue,
                "foo_score": mock_issues.foo_score,
                "is_label_issue": [False, False, False, False, False],
                "label_score": [0.95071431, 0.15601864, 0.60111501, 0.70807258, 0.18182497],
            }
        )
        pd.testing.assert_frame_equal(lab.issues, expected_issues_df, check_exact=False)

        expected_issue_summary_df = pd.DataFrame(
            {
                "issue_type": ["foo", "label"],
                "score": [0.72, 0.6],
            }
        )
        pd.testing.assert_frame_equal(
            lab.issue_summary, expected_issue_summary_df, check_exact=False
        )

    def test_save(self, lab, tmp_path, monkeypatch):
        """Test that the save and load methods work."""
        lab.save(tmp_path)
        assert tmp_path.exists(), "Save directory was not created"
        assert (tmp_path / "data").is_dir(), "Data directory was not saved"
        assert (tmp_path / "issues.csv").exists(), "Issues file was not saved"
        assert (tmp_path / "summary.csv").exists(), "Issue summary file was not saved"
        assert (tmp_path / "datalab.pkl").exists(), "Datalab file was not saved"

        # Mock the issues dataframe
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        # Mock the issue summary dataframe
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)
        lab.save(tmp_path)
        assert (tmp_path / "issues.csv").exists(), "Issues file was not saved"
        assert (tmp_path / "summary.csv").exists(), "Issue summary file was not saved"

        # Save works in an arbitrary directory, that should be created if it doesn't exist
        new_dir = tmp_path / "subdir"
        assert not new_dir.exists(), "Directory should not exist"
        lab.save(new_dir)
        assert new_dir.exists(), "Directory was not created"

    def test_pickle(self, lab, tmp_path):
        """Test that the class can be pickled."""
        pickle_file = os.path.join(tmp_path, "lab.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(lab, f)
        with open(pickle_file, "rb") as f:
            lab2 = pickle.load(f)

        assert lab2.label_name == "star"

    def test_load(self, lab, tmp_path, dataset, monkeypatch):
        """Test that the save and load methods work."""

        # Mock the issues dataframe
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        # Mock the issue summary dataframe
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        lab.save(tmp_path)

        loaded_lab = Datalab.load(tmp_path)
        data = lab._data
        loaded_data = loaded_lab._data
        assert loaded_data == data
        assert loaded_lab.info == lab.info
        pd.testing.assert_frame_equal(loaded_lab.issues, mock_issues)
        pd.testing.assert_frame_equal(loaded_lab.issue_summary, mock_issue_summary)

        # Load accepts a `Dataset`.
        loaded_lab = Datalab.load(tmp_path, data=dataset)
        assert loaded_lab.data._data == dataset.data

        # Misaligned dataset raises a ValueError
        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset.shard(2, 0))
            expected_error_msg = "Length of data (2) does not match length of labels (5)"
            assert expected_error_msg == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset.shuffle())
            expected_error_msg = (
                "Data has been modified since Lab was saved. Cannot load Lab with modified data."
            )
            assert expected_error_msg == str(excinfo.value)

    def test_failed_issue_managers(self, lab, monkeypatch):
        """Test that a failed issue manager will not be added to the Datalab instance after
        the call to `find_issues`."""
        mock_issue_types = {"erroneous_issue_type": {}}
        monkeypatch.setattr(lab, "_set_issue_types", lambda *args, **kwargs: mock_issue_types)

        mock_issue_manager = Mock()
        mock_issue_manager.name = "erronous"
        mock_issue_manager.find_issues.side_effect = ValueError("Some error")

        class MockIssueManagerFactory:
            @staticmethod
            def from_list(*args, **kwargs):
                return [mock_issue_manager]

        monkeypatch.setattr(
            "cleanlab.experimental.datalab.datalab._IssueManagerFactory", MockIssueManagerFactory
        )

        assert lab.issues.empty
        lab.find_issues()
        assert lab.issues.empty

    @pytest.mark.parametrize("include_description", [True, False])
    def test_get_report(self, lab, include_description, monkeypatch):
        """Test that the report method works. Assuming we have two issue managers, each should add
        their section to the report."""

        mock_issue_manager = Mock()
        mock_issue_manager.issue_name = "foo"
        mock_issue_manager.report.return_value = "foo report"

        class MockIssueManagerFactory:
            @staticmethod
            def from_str(*args, **kwargs):
                return mock_issue_manager

        monkeypatch.setattr(
            "cleanlab.experimental.datalab.datalab._IssueManagerFactory", MockIssueManagerFactory
        )
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )

        mock_info = {"foo": {"bar": "baz"}}

        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        monkeypatch.setattr(
            lab, "_add_issue_summary_to_report", lambda *args, **kwargs: "Here is a lab summary\n\n"
        )
        monkeypatch.setattr(lab, "issues", mock_issues, raising=False)
        monkeypatch.setattr(lab, "info", mock_info, raising=False)

        report = lab._get_report(k=3, verbosity=0, include_description=include_description)
        expected_report = "\n\n".join(["Here is a lab summary", "foo report"])
        assert report == expected_report

    def test_report(self, lab, monkeypatch):
        # lab.report simply wraps _get_report in a print statement
        mock_get_report = Mock()
        # Define a mock function that takes a verbosity argument and a k argument
        # and returns a string
        mock_get_report.side_effect = (
            lambda verbosity, k, **kwargs: f"Report with verbosity={verbosity} and k={k}"
        )
        monkeypatch.setattr(lab, "_get_report", mock_get_report)

        # Call report with no arguments, test that it prints the report
        with patch("builtins.print") as mock_print:
            lab.report()
            mock_print.assert_called_once_with("Report with verbosity=0 and k=5")
            mock_print.reset_mock()
            lab.report(k=10, verbosity=3)
            mock_print.assert_called_once_with("Report with verbosity=3 and k=10")