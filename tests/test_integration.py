"""Integration tests for the complete ML pipeline."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.svm import SVC

from src.color_correlogram import extract_correlogram_feature
from src.color_histogram import extract_histogram_feature
from src.preprocessing import convert_to_hsv, load_image, quantize_colors_hsv
from src.evaluation_methods import evaluate_stratified_holdout


class TestEndToEndPipeline:
    """End-to-end integration tests for the ML pipeline."""

    def test_pipeline_image_to_feature_correlogram(self, sample_image):
        """Test complete pipeline: image -> preprocessing -> feature extraction."""
        # Step 1: Convert to HSV
        img_hsv = convert_to_hsv(sample_image)
        assert img_hsv is not None

        # Step 2: Extract correlogram feature
        feature = extract_correlogram_feature(sample_image, color_space="hsv")
        assert feature is not None
        assert feature.shape == (72 * 4,)
        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)

    def test_pipeline_image_to_feature_histogram(self, sample_image):
        """Test complete pipeline: image -> preprocessing -> histogram extraction."""
        # Step 1: Extract histogram feature
        feature = extract_histogram_feature(sample_image, color_space="hsv")
        assert feature is not None
        assert feature.shape == (72,)
        assert np.isclose(feature.sum(), 1.0)

    def test_pipeline_reproducibility_same_seed(self, sample_image):
        """Test pipeline produces identical results with same seed."""
        feature1 = extract_correlogram_feature(sample_image)
        feature2 = extract_correlogram_feature(sample_image)

        np.testing.assert_array_equal(feature1, feature2)

    def test_pipeline_feature_extraction_consistency(self, sample_image):
        """Test feature extraction is consistent across multiple calls."""
        features = [extract_correlogram_feature(sample_image) for _ in range(3)]

        for i in range(1, len(features)):
            np.testing.assert_array_equal(features[0], features[i])

    def test_pipeline_different_images_different_features(self):
        """Test different images produce different features."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        feat1 = extract_correlogram_feature(img1)
        feat2 = extract_correlogram_feature(img2)

        # Features should be different (with very high probability)
        assert not np.allclose(feat1, feat2)

    def test_pipeline_evaluation_with_features(self, sample_dataset):
        """Test evaluation pipeline with extracted features."""
        X, y = sample_dataset

        def train_fn(X_train, y_train):
            model = SVC(kernel="linear", random_state=42)
            model.fit(X_train, y_train)
            return model, {"epochs": 1}

        y_test, y_pred, info = evaluate_stratified_holdout(train_fn, X, y)

        assert y_test is not None
        assert y_pred is not None
        assert len(y_test) == len(y_pred)
        assert info["method"] == "stratified_holdout"

    def test_pipeline_multiple_color_spaces(self, sample_image):
        """Test pipeline works with different color spaces."""
        feat_hsv = extract_correlogram_feature(sample_image, color_space="hsv")
        feat_rgb = extract_correlogram_feature(sample_image, color_space="rgb")

        assert feat_hsv is not None
        assert feat_rgb is not None
        # Different color spaces have different feature dimensions
        # HSV: 8*3*3*4 = 288, RGB: 4*4*4*4 = 256
        assert feat_hsv.shape[0] != feat_rgb.shape[0]

    def test_pipeline_spatial_features(self, sample_image):
        """Test pipeline with spatial grid features."""
        feat_global = extract_correlogram_feature(
            sample_image, color_space="hsv", spatial_grid=None
        )
        feat_spatial = extract_correlogram_feature(
            sample_image, color_space="hsv", spatial_grid=2
        )

        assert feat_global is not None
        assert feat_spatial is not None
        # Spatial features should be larger (global + patches)
        assert feat_spatial.shape[0] > feat_global.shape[0]

    def test_pipeline_batch_processing(self, sample_image):
        """Test processing multiple images in batch."""
        images = [sample_image for _ in range(5)]
        features = [extract_correlogram_feature(img) for img in images]

        assert len(features) == 5
        for feat in features:
            assert feat.shape == features[0].shape

    def test_pipeline_deterministic_with_seed(self):
        """Test pipeline determinism across different random images."""
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        np.random.seed(42)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        feat1 = extract_correlogram_feature(img1)
        feat2 = extract_correlogram_feature(img2)

        np.testing.assert_array_equal(feat1, feat2)


class TestPipelineWithRealImages:
    """Integration tests with real image files."""

    def test_pipeline_load_and_extract(self, sample_image_file):
        """Test loading image from file and extracting features."""
        img = load_image(sample_image_file)
        assert img is not None

        feature = extract_correlogram_feature(img)
        assert feature is not None
        assert feature.shape == (72 * 4,)

    def test_pipeline_multiple_sizes(self, temp_dir, sample_image):
        """Test pipeline with different image sizes."""
        sizes = [(64, 64), (128, 128), (256, 256)]

        for size in sizes:
            img_path = Path(temp_dir) / f"img_{size[0]}.jpg"
            resized = cv2.resize(sample_image, size)
            cv2.imwrite(str(img_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

            img = load_image(img_path, size=size)
            feature = extract_correlogram_feature(img)

            assert feature is not None
            assert feature.shape == (72 * 4,)
