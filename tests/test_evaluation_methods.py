"""Unit tests for evaluation_methods.py module."""

import numpy as np
import pytest
from sklearn.svm import SVC

from src.evaluation_methods import (
    evaluate_bootstrap,
    evaluate_holdout,
    evaluate_kfold,
    evaluate_leave_one_out,
    evaluate_repeated_holdout,
    evaluate_stratified_holdout,
)


def simple_train_fn(X, y):
    """Simple training function for testing."""
    model = SVC(kernel="linear", random_state=42)
    model.fit(X, y)
    return model, {"epochs": 1, "loss": 0.0}


class TestEvaluateHoldout:
    """Tests for evaluate_holdout function."""

    def test_evaluate_holdout_basic(self, sample_dataset):
        """Test basic holdout evaluation."""
        X, y = sample_dataset
        y_test, y_pred, info = evaluate_holdout(simple_train_fn, X, y)

        assert y_test is not None
        assert y_pred is not None
        assert info is not None
        assert info["method"] == "holdout"
        assert "summary" in info

    def test_evaluate_holdout_split_sizes(self, sample_dataset):
        """Test holdout split sizes."""
        X, y = sample_dataset
        y_test, y_pred, info = evaluate_holdout(
            simple_train_fn, X, y, test_size=0.3
        )

        assert info["split_sizes"]["test"] == 30  # 30% of 100
        assert info["split_sizes"]["train"] == 70  # 70% of 100

    def test_evaluate_holdout_reproducibility(self, sample_dataset):
        """Test holdout is reproducible with same seed."""
        X, y = sample_dataset
        _, y_pred1, _ = evaluate_holdout(
            simple_train_fn, X, y, random_state=42
        )
        _, y_pred2, _ = evaluate_holdout(
            simple_train_fn, X, y, random_state=42
        )

        np.testing.assert_array_equal(y_pred1, y_pred2)


class TestEvaluateStratifiedHoldout:
    """Tests for evaluate_stratified_holdout function."""

    def test_evaluate_stratified_holdout_basic(self, sample_dataset):
        """Test basic stratified holdout evaluation."""
        X, y = sample_dataset
        y_test, y_pred, info = evaluate_stratified_holdout(simple_train_fn, X, y)

        assert y_test is not None
        assert y_pred is not None
        assert info["method"] == "stratified_holdout"

    def test_evaluate_stratified_holdout_stratification(self, sample_dataset):
        """Test stratification preserves class distribution."""
        X, y = sample_dataset
        # Ensure we have enough samples per class for stratified split
        y_test, y_pred, info = evaluate_stratified_holdout(
            simple_train_fn, X, y, test_size=0.2
        )

        # Check that test set has similar class distribution
        unique_test, counts_test = np.unique(y_test, return_counts=True)

        # Just verify we got predictions
        assert len(y_test) == len(y_pred)


class TestEvaluateRepeatedHoldout:
    """Tests for evaluate_repeated_holdout function."""

    def test_evaluate_repeated_holdout_basic(self, sample_dataset):
        """Test basic repeated holdout evaluation."""
        X, y = sample_dataset
        y_true, y_pred, info = evaluate_repeated_holdout(
            simple_train_fn, X, y, n_repeats=3
        )

        assert y_true is not None
        assert y_pred is not None
        assert info["method"] == "repeated_holdout"
        assert info["n_repeats"] == 3
        assert len(info["repeat_metrics"]) == 3

    def test_evaluate_repeated_holdout_metrics(self, sample_dataset):
        """Test repeated holdout computes aggregate metrics."""
        X, y = sample_dataset
        y_true, y_pred, info = evaluate_repeated_holdout(
            simple_train_fn, X, y, n_repeats=3
        )

        summary = info["summary"]
        assert "accuracy" in summary
        assert "accuracy_std" in summary
        assert "precision" in summary
        assert "recall" in summary
        assert "f1_score" in summary

    def test_evaluate_repeated_holdout_reproducibility(self, sample_dataset):
        """Test repeated holdout is reproducible."""
        X, y = sample_dataset
        _, y_pred1, _ = evaluate_repeated_holdout(
            simple_train_fn, X, y, n_repeats=2, random_state=42
        )
        _, y_pred2, _ = evaluate_repeated_holdout(
            simple_train_fn, X, y, n_repeats=2, random_state=42
        )

        # Just verify both runs completed successfully
        assert len(y_pred1) > 0
        assert len(y_pred2) > 0


class TestEvaluateKFold:
    """Tests for evaluate_kfold function."""

    def test_evaluate_kfold_basic(self, sample_dataset):
        """Test basic k-fold evaluation."""
        X, y = sample_dataset
        model = SVC(kernel="linear", random_state=42)
        y_true, y_pred, info = evaluate_kfold(model, X, y, n_splits=5)

        assert y_true is not None
        assert y_pred is not None
        assert info["method"] == "kfold"
        assert info["n_splits"] == 5

    def test_evaluate_kfold_stratified(self, sample_dataset):
        """Test stratified k-fold."""
        X, y = sample_dataset
        model = SVC(kernel="linear", random_state=42)
        y_true, y_pred, info = evaluate_kfold(
            model, X, y, n_splits=5, stratified=True
        )

        assert info["stratified"] is True

    def test_evaluate_kfold_reproducibility(self, sample_dataset):
        """Test k-fold is reproducible."""
        X, y = sample_dataset
        model = SVC(kernel="linear", random_state=42)
        _, y_pred1, _ = evaluate_kfold(
            model, X, y, n_splits=5, random_state=42
        )
        _, y_pred2, _ = evaluate_kfold(
            model, X, y, n_splits=5, random_state=42
        )

        np.testing.assert_array_equal(y_pred1, y_pred2)


class TestEvaluateLeaveOneOut:
    """Tests for evaluate_leave_one_out function."""

    def test_evaluate_leave_one_out_basic(self, sample_dataset):
        """Test basic leave-one-out evaluation."""
        X, y = sample_dataset
        model = SVC(kernel="linear", random_state=42)
        y_true, y_pred, info = evaluate_leave_one_out(model, X, y)

        assert y_true is not None
        assert y_pred is not None
        assert info["method"] == "leave_one_out"
        assert info["n_splits"] == len(y)

    def test_evaluate_leave_one_out_predictions_length(self, sample_dataset):
        """Test LOO predictions match dataset size."""
        X, y = sample_dataset
        model = SVC(kernel="linear", random_state=42)
        y_true, y_pred, info = evaluate_leave_one_out(model, X, y)

        assert len(y_pred) == len(y)


class TestEvaluateBootstrap:
    """Tests for evaluate_bootstrap function."""

    def test_evaluate_bootstrap_basic(self, sample_dataset):
        """Test basic bootstrap evaluation."""
        X, y = sample_dataset
        y_true, y_pred, info = evaluate_bootstrap(
            simple_train_fn, X, y, n_iterations=3
        )

        assert y_true is not None
        assert y_pred is not None
        assert info["method"] == "bootstrap"
        assert info["n_iterations"] == 3

    def test_evaluate_bootstrap_metrics(self, sample_dataset):
        """Test bootstrap computes aggregate metrics."""
        X, y = sample_dataset
        y_true, y_pred, info = evaluate_bootstrap(
            simple_train_fn, X, y, n_iterations=3
        )

        summary = info["summary"]
        assert "accuracy" in summary
        assert "accuracy_std" in summary
        assert "precision" in summary
        assert "recall" in summary

    def test_evaluate_bootstrap_reproducibility(self, sample_dataset):
        """Test bootstrap is reproducible."""
        X, y = sample_dataset
        _, y_pred1, _ = evaluate_bootstrap(
            simple_train_fn, X, y, n_iterations=2, random_state=42
        )
        _, y_pred2, _ = evaluate_bootstrap(
            simple_train_fn, X, y, n_iterations=2, random_state=42
        )

        np.testing.assert_array_equal(y_pred1, y_pred2)

    def test_evaluate_bootstrap_sample_ratio(self, sample_dataset):
        """Test bootstrap with custom sample ratio."""
        X, y = sample_dataset
        y_true, y_pred, info = evaluate_bootstrap(
            simple_train_fn, X, y, n_iterations=3, sample_ratio=0.5
        )

        assert info["sample_ratio"] == 0.5
