"""Unit tests for color_correlogram.py module."""

import numpy as np
import pytest

from src.color_correlogram import (
    auto_correlogram,
    auto_correlogram_fast,
    extract_correlogram_feature,
    spatial_correlogram,
)


class TestAutoCorrelogram:
    """Tests for auto_correlogram function."""

    def test_auto_correlogram_basic(self):
        """Test basic auto_correlogram computation."""
        # Create simple quantized image
        quantized = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
        n_colors = 2

        feature = auto_correlogram(quantized, n_colors)

        assert feature is not None
        assert isinstance(feature, np.ndarray)
        assert feature.dtype == np.float64
        # Default distances [1, 3, 5, 7], so feature size = n_colors * len(distances)
        assert feature.shape == (2 * 4,)

    def test_auto_correlogram_custom_distances(self):
        """Test auto_correlogram with custom distances."""
        quantized = np.random.randint(0, 8, (50, 50), dtype=np.int32)
        n_colors = 8
        distances = [1, 2]

        feature = auto_correlogram(quantized, n_colors, distances=distances)

        assert feature.shape == (n_colors * len(distances),)

    def test_auto_correlogram_value_range(self):
        """Test auto_correlogram values are in [0, 1]."""
        quantized = np.random.randint(0, 16, (100, 100), dtype=np.int32)
        n_colors = 16

        feature = auto_correlogram(quantized, n_colors)

        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)

    def test_auto_correlogram_single_color(self):
        """Test auto_correlogram with single color image."""
        quantized = np.zeros((50, 50), dtype=np.int32)
        n_colors = 1

        feature = auto_correlogram(quantized, n_colors)

        # All pixels same color, so correlogram should be 1.0 for that color
        assert feature[0] == 1.0


class TestAutoCorrelogramFast:
    """Tests for auto_correlogram_fast function."""

    def test_auto_correlogram_fast_basic(self):
        """Test fast auto_correlogram computation."""
        quantized = np.random.randint(0, 8, (50, 50), dtype=np.int32)
        n_colors = 8

        feature = auto_correlogram_fast(quantized, n_colors)

        assert feature is not None
        assert isinstance(feature, np.ndarray)
        assert feature.shape == (n_colors * 4,)  # Default 4 distances

    def test_auto_correlogram_fast_vs_slow(self):
        """Test fast version produces similar results to slow version."""
        quantized = np.random.randint(0, 8, (50, 50), dtype=np.int32)
        n_colors = 8
        distances = [1, 3]

        feat_slow = auto_correlogram(quantized, n_colors, distances=distances)
        feat_fast = auto_correlogram_fast(quantized, n_colors, distances=distances)

        # Results should be very close (allowing for numerical precision)
        np.testing.assert_allclose(feat_slow, feat_fast, rtol=1e-10)

    def test_auto_correlogram_fast_value_range(self):
        """Test fast version values are in [0, 1]."""
        quantized = np.random.randint(0, 16, (100, 100), dtype=np.int32)
        n_colors = 16

        feature = auto_correlogram_fast(quantized, n_colors)

        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)


class TestSpatialCorrelogram:
    """Tests for spatial_correlogram function."""

    def test_spatial_correlogram_basic(self):
        """Test spatial correlogram computation."""
        quantized = np.random.randint(0, 8, (100, 100), dtype=np.int32)
        n_colors = 8
        grid_size = 2

        feature = spatial_correlogram(quantized, n_colors, grid_size=grid_size)

        assert feature is not None
        assert isinstance(feature, np.ndarray)
        # Global + grid_size^2 patches, each with n_colors * 4 dimensions
        expected_size = (1 + grid_size * grid_size) * n_colors * 4
        assert feature.shape == (expected_size,)

    def test_spatial_correlogram_without_global(self):
        """Test spatial correlogram without global feature."""
        quantized = np.random.randint(0, 8, (100, 100), dtype=np.int32)
        n_colors = 8
        grid_size = 2

        feature = spatial_correlogram(
            quantized, n_colors, grid_size=grid_size, include_global=False
        )

        # Only grid patches, no global
        expected_size = grid_size * grid_size * n_colors * 4
        assert feature.shape == (expected_size,)

    def test_spatial_correlogram_value_range(self):
        """Test spatial correlogram values are in [0, 1]."""
        quantized = np.random.randint(0, 16, (100, 100), dtype=np.int32)
        n_colors = 16

        feature = spatial_correlogram(quantized, n_colors, grid_size=2)

        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)


class TestExtractCorrelogramFeature:
    """Tests for extract_correlogram_feature function."""

    def test_extract_correlogram_feature_hsv(self, sample_image):
        """Test extracting correlogram feature from HSV image."""
        feature = extract_correlogram_feature(sample_image, color_space="hsv")

        assert feature is not None
        assert isinstance(feature, np.ndarray)
        # Default: 8*3*3 colors * 4 distances
        assert feature.shape == (72 * 4,)

    def test_extract_correlogram_feature_rgb(self, sample_image):
        """Test extracting correlogram feature from RGB image."""
        feature = extract_correlogram_feature(sample_image, color_space="rgb")

        assert feature is not None
        # Default: 4^3 colors * 4 distances
        assert feature.shape == (64 * 4,)

    def test_extract_correlogram_feature_custom_bins(self, sample_image):
        """Test extracting correlogram with custom bins."""
        feature = extract_correlogram_feature(
            sample_image, color_space="hsv", h_bins=4, s_bins=2, v_bins=2
        )

        # 4*2*2 colors * 4 distances
        assert feature.shape == (16 * 4,)

    def test_extract_correlogram_feature_spatial(self, sample_image):
        """Test extracting correlogram with spatial grid."""
        feature = extract_correlogram_feature(
            sample_image, color_space="hsv", spatial_grid=2
        )

        # (1 global + 2*2 patches) * 72 colors * 4 distances
        assert feature.shape == (5 * 72 * 4,)

    def test_extract_correlogram_feature_invalid_color_space(self, sample_image):
        """Test invalid color space raises error."""
        with pytest.raises(ValueError, match="color_space"):
            extract_correlogram_feature(sample_image, color_space="invalid")

    def test_extract_correlogram_feature_value_range(self, sample_image):
        """Test extracted feature values are in valid range."""
        feature = extract_correlogram_feature(sample_image)

        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)

    def test_extract_correlogram_feature_deterministic(self, sample_image):
        """Test feature extraction is deterministic."""
        feature1 = extract_correlogram_feature(sample_image)
        feature2 = extract_correlogram_feature(sample_image)

        np.testing.assert_array_equal(feature1, feature2)


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_quantized_image_invalid_type(self):
        """Test validation rejects non-array input."""
        with pytest.raises(TypeError):
            auto_correlogram("not an array", 8)

    def test_validate_quantized_image_invalid_ndim(self):
        """Test validation rejects wrong dimensions."""
        with pytest.raises(ValueError):
            auto_correlogram(np.zeros((10, 10, 3), dtype=np.int32), 8)

    def test_validate_quantized_image_invalid_dtype(self):
        """Test validation rejects wrong dtype."""
        with pytest.raises(TypeError):
            auto_correlogram(np.zeros((10, 10), dtype=np.float32), 8)

    def test_validate_n_colors_invalid_type(self):
        """Test validation rejects invalid n_colors type."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, "not a number")

    def test_validate_n_colors_zero(self):
        """Test validation rejects zero n_colors."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, 0)

    def test_validate_n_colors_negative(self):
        """Test validation rejects negative n_colors."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, -5)

    def test_validate_distances_invalid_type(self):
        """Test validation rejects invalid distances type."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(TypeError):
            auto_correlogram(quantized, 8, distances="not a list")

    def test_validate_distances_empty_list(self):
        """Test validation rejects empty distances list."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, 8, distances=[])

    def test_validate_distances_invalid_values(self):
        """Test validation rejects invalid distance values."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, 8, distances=[0, 1, 2])

    def test_validate_distances_negative_values(self):
        """Test validation rejects negative distances."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            auto_correlogram(quantized, 8, distances=[-1, 1, 2])


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_auto_correlogram_tiny_image(self):
        """Test auto_correlogram with 1x1 image."""
        quantized = np.array([[0]], dtype=np.int32)
        feature = auto_correlogram(quantized, 1)
        assert feature is not None
        assert feature.shape == (1 * 4,)

    def test_auto_correlogram_single_pixel(self):
        """Test auto_correlogram with single pixel."""
        quantized = np.array([[5]], dtype=np.int32)
        feature = auto_correlogram(quantized, 10)
        assert feature is not None

    def test_auto_correlogram_large_distance(self):
        """Test auto_correlogram with large distance values."""
        quantized = np.random.randint(0, 8, (100, 100), dtype=np.int32)
        feature = auto_correlogram(quantized, 8, distances=[50, 100])
        assert feature is not None
        assert feature.shape == (8 * 2,)

    def test_spatial_correlogram_grid_size_1(self):
        """Test spatial_correlogram with grid_size=1."""
        quantized = np.random.randint(0, 8, (100, 100), dtype=np.int32)
        feature = spatial_correlogram(quantized, 8, grid_size=1)
        assert feature is not None
        # 1 global + 1*1 patch = 2 patches
        assert feature.shape == (2 * 8 * 4,)

    def test_spatial_correlogram_large_grid(self):
        """Test spatial_correlogram with large grid_size."""
        quantized = np.random.randint(0, 8, (200, 200), dtype=np.int32)
        feature = spatial_correlogram(quantized, 8, grid_size=4)
        assert feature is not None
        # 1 global + 4*4 patches = 17 patches
        assert feature.shape == (17 * 8 * 4,)

    def test_extract_correlogram_feature_extreme_bins(self, sample_image):
        """Test extracting correlogram with extreme bin values."""
        feature = extract_correlogram_feature(
            sample_image, color_space="hsv", h_bins=16, s_bins=8, v_bins=8
        )
        assert feature is not None
        assert feature.shape == (16 * 8 * 8 * 4,)

    def test_extract_correlogram_feature_single_bin(self, sample_image):
        """Test extracting correlogram with single bin."""
        feature = extract_correlogram_feature(
            sample_image, color_space="hsv", h_bins=1, s_bins=1, v_bins=1
        )
        assert feature is not None
        assert feature.shape == (1 * 4,)
