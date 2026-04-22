"""Unit tests for color_histogram.py module."""

import numpy as np
import pytest

from src.color_histogram import color_histogram, extract_histogram_feature


class TestColorHistogram:
    """Tests for color_histogram function."""

    def test_color_histogram_basic(self):
        """Test basic color histogram computation."""
        quantized = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
        n_colors = 2

        histogram = color_histogram(quantized, n_colors)

        assert histogram is not None
        assert isinstance(histogram, np.ndarray)
        assert histogram.shape == (n_colors,)
        assert histogram.dtype == np.float64

    def test_color_histogram_sums_to_one(self):
        """Test histogram sums to 1.0."""
        quantized = np.random.randint(0, 16, (100, 100), dtype=np.int32)
        n_colors = 16

        histogram = color_histogram(quantized, n_colors)

        assert np.isclose(histogram.sum(), 1.0)

    def test_color_histogram_value_range(self):
        """Test histogram values are in [0, 1]."""
        quantized = np.random.randint(0, 8, (50, 50), dtype=np.int32)
        n_colors = 8

        histogram = color_histogram(quantized, n_colors)

        assert np.all(histogram >= 0.0)
        assert np.all(histogram <= 1.0)

    def test_color_histogram_single_color(self):
        """Test histogram with single color image."""
        quantized = np.zeros((50, 50), dtype=np.int32)
        n_colors = 1

        histogram = color_histogram(quantized, n_colors)

        assert histogram[0] == 1.0

    def test_color_histogram_uniform_distribution(self):
        """Test histogram with uniform color distribution."""
        # Create image with equal number of each color
        quantized = np.array([[0, 1], [2, 3]], dtype=np.int32)
        n_colors = 4

        histogram = color_histogram(quantized, n_colors)

        # Each color appears once, so histogram should be [0.25, 0.25, 0.25, 0.25]
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(histogram, expected)

    def test_color_histogram_empty_colors(self):
        """Test histogram with unused colors."""
        quantized = np.array([[0, 0], [0, 0]], dtype=np.int32)
        n_colors = 5

        histogram = color_histogram(quantized, n_colors)

        # Only color 0 is used
        assert histogram[0] == 1.0
        assert np.all(histogram[1:] == 0.0)


class TestExtractHistogramFeature:
    """Tests for extract_histogram_feature function."""

    def test_extract_histogram_feature_hsv(self, sample_image):
        """Test extracting histogram feature from HSV image."""
        feature = extract_histogram_feature(sample_image, color_space="hsv")

        assert feature is not None
        assert isinstance(feature, np.ndarray)
        # Default: 8*3*3 colors
        assert feature.shape == (72,)

    def test_extract_histogram_feature_rgb(self, sample_image):
        """Test extracting histogram feature from RGB image."""
        feature = extract_histogram_feature(sample_image, color_space="rgb")

        assert feature is not None
        # Default: 4^3 colors
        assert feature.shape == (64,)

    def test_extract_histogram_feature_sums_to_one(self, sample_image):
        """Test extracted histogram sums to 1.0."""
        feature = extract_histogram_feature(sample_image)

        assert np.isclose(feature.sum(), 1.0)

    def test_extract_histogram_feature_value_range(self, sample_image):
        """Test extracted feature values are in [0, 1]."""
        feature = extract_histogram_feature(sample_image)

        assert np.all(feature >= 0.0)
        assert np.all(feature <= 1.0)

    def test_extract_histogram_feature_custom_bins(self, sample_image):
        """Test extracting histogram with custom bins."""
        feature = extract_histogram_feature(
            sample_image, color_space="hsv", h_bins=4, s_bins=2, v_bins=2
        )

        # 4*2*2 colors
        assert feature.shape == (16,)

    def test_extract_histogram_feature_invalid_color_space(self, sample_image):
        """Test invalid color space raises error."""
        with pytest.raises(ValueError, match="color_space"):
            extract_histogram_feature(sample_image, color_space="invalid")

    def test_extract_histogram_feature_deterministic(self, sample_image):
        """Test feature extraction is deterministic."""
        feature1 = extract_histogram_feature(sample_image)
        feature2 = extract_histogram_feature(sample_image)

        np.testing.assert_array_equal(feature1, feature2)


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_quantized_image_invalid_type(self):
        """Test validation rejects non-array input."""
        with pytest.raises(TypeError):
            color_histogram("not an array", 8)

    def test_validate_quantized_image_invalid_ndim(self):
        """Test validation rejects wrong dimensions."""
        with pytest.raises(ValueError):
            color_histogram(np.zeros((10, 10, 3), dtype=np.int32), 8)

    def test_validate_n_colors_invalid_type(self):
        """Test validation rejects invalid n_colors type."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            color_histogram(quantized, "not a number")

    def test_validate_n_colors_zero(self):
        """Test validation rejects zero n_colors."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            color_histogram(quantized, 0)

    def test_validate_n_colors_negative(self):
        """Test validation rejects negative n_colors."""
        quantized = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            color_histogram(quantized, -5)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_color_histogram_tiny_image(self):
        """Test color_histogram with 1x1 image."""
        quantized = np.array([[0]], dtype=np.int32)
        histogram = color_histogram(quantized, 1)
        assert histogram is not None
        assert histogram.shape == (1,)
        assert histogram[0] == 1.0

    def test_color_histogram_single_pixel(self):
        """Test color_histogram with single pixel."""
        quantized = np.array([[5]], dtype=np.int32)
        histogram = color_histogram(quantized, 10)
        assert histogram is not None
        assert histogram[5] == 1.0

    def test_extract_histogram_feature_extreme_bins(self, sample_image):
        """Test extracting histogram with extreme bin values."""
        feature = extract_histogram_feature(
            sample_image, color_space="hsv", h_bins=16, s_bins=8, v_bins=8
        )
        assert feature is not None
        assert feature.shape == (16 * 8 * 8,)
        assert np.isclose(feature.sum(), 1.0)

    def test_extract_histogram_feature_single_bin(self, sample_image):
        """Test extracting histogram with single bin."""
        feature = extract_histogram_feature(
            sample_image, color_space="hsv", h_bins=1, s_bins=1, v_bins=1
        )
        assert feature is not None
        assert feature.shape == (1,)
        assert np.isclose(feature.sum(), 1.0)

    def test_extract_histogram_feature_rgb_extreme_bins(self, sample_image):
        """Test extracting RGB histogram with extreme bins."""
        feature = extract_histogram_feature(
            sample_image, color_space="rgb", rgb_bins=16
        )
        assert feature is not None
        assert feature.shape == (16 ** 3,)
        assert np.isclose(feature.sum(), 1.0)

    def test_color_histogram_large_n_colors(self):
        """Test color_histogram with large n_colors."""
        quantized = np.random.randint(0, 256, (100, 100), dtype=np.int32)
        histogram = color_histogram(quantized, 256)
        assert histogram is not None
        assert histogram.shape == (256,)
        assert np.isclose(histogram.sum(), 1.0)

    def test_extract_histogram_feature_rgb_custom_bins(self, sample_image):
        """Test extracting RGB histogram with custom bins."""
        feature = extract_histogram_feature(
            sample_image, color_space="rgb", rgb_bins=8
        )
        assert feature is not None
        assert feature.shape == (8 ** 3,)
        assert np.isclose(feature.sum(), 1.0)
