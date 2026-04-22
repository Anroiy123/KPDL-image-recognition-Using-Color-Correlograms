"""Unit tests for preprocessing.py module."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    convert_to_hsv,
    get_class_names,
    load_dataset,
    load_image,
    quantize_colors_hsv,
    quantize_colors_rgb,
)


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_image_valid(self, sample_image_file):
        """Test loading a valid image file."""
        img = load_image(sample_image_file)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.shape == (128, 128, 3)

    def test_load_image_custom_size(self, sample_image_file):
        """Test loading image with custom size."""
        img = load_image(sample_image_file, size=(64, 64))
        assert img.shape == (64, 64, 3)

    def test_load_image_nonexistent(self):
        """Test loading nonexistent image raises error."""
        with pytest.raises(ValueError, match="Khong the doc anh"):
            load_image("/nonexistent/path/image.jpg")

    def test_load_image_invalid_path_type(self):
        """Test loading with invalid path type."""
        with pytest.raises((ValueError, TypeError)):
            load_image(None)


class TestConvertToHSV:
    """Tests for convert_to_hsv function."""

    def test_convert_to_hsv_valid(self, sample_image):
        """Test converting BGR image to HSV."""
        hsv = convert_to_hsv(sample_image)
        assert hsv is not None
        assert isinstance(hsv, np.ndarray)
        assert hsv.shape == sample_image.shape
        assert hsv.dtype == np.uint8

    def test_convert_to_hsv_value_range(self, sample_image):
        """Test HSV values are in valid range."""
        hsv = convert_to_hsv(sample_image)
        assert np.all(hsv[:, :, 0] <= 179)  # Hue in OpenCV
        assert np.all(hsv[:, :, 1] <= 255)  # Saturation
        assert np.all(hsv[:, :, 2] <= 255)  # Value


class TestQuantizeColorsHSV:
    """Tests for quantize_colors_hsv function."""

    def test_quantize_colors_hsv_basic(self, sample_image):
        """Test basic HSV quantization."""
        img_hsv = convert_to_hsv(sample_image)
        quantized, n_colors = quantize_colors_hsv(img_hsv)

        assert quantized is not None
        assert isinstance(quantized, np.ndarray)
        assert quantized.shape == (100, 100)
        assert n_colors == 8 * 3 * 3  # Default bins
        assert np.all(quantized >= 0)
        assert np.all(quantized < n_colors)

    def test_quantize_colors_hsv_custom_bins(self, sample_image):
        """Test HSV quantization with custom bins."""
        img_hsv = convert_to_hsv(sample_image)
        quantized, n_colors = quantize_colors_hsv(img_hsv, h_bins=4, s_bins=2, v_bins=2)

        assert n_colors == 4 * 2 * 2
        assert np.all(quantized < n_colors)

    def test_quantize_colors_hsv_output_range(self, sample_image):
        """Test quantized values are in valid range."""
        img_hsv = convert_to_hsv(sample_image)
        quantized, n_colors = quantize_colors_hsv(img_hsv)

        assert quantized.min() >= 0
        assert quantized.max() < n_colors


class TestQuantizeColorsRGB:
    """Tests for quantize_colors_rgb function."""

    def test_quantize_colors_rgb_basic(self, sample_image):
        """Test basic RGB quantization."""
        quantized, n_colors = quantize_colors_rgb(sample_image)

        assert quantized is not None
        assert isinstance(quantized, np.ndarray)
        assert quantized.shape == (100, 100)
        assert n_colors == 4 ** 3  # Default bins=4
        assert np.all(quantized >= 0)
        assert np.all(quantized < n_colors)

    def test_quantize_colors_rgb_custom_bins(self, sample_image):
        """Test RGB quantization with custom bins."""
        quantized, n_colors = quantize_colors_rgb(sample_image, bins=8)

        assert n_colors == 8 ** 3
        assert np.all(quantized < n_colors)

    def test_quantize_colors_rgb_output_range(self, sample_image):
        """Test quantized values are in valid range."""
        quantized, n_colors = quantize_colors_rgb(sample_image)

        assert quantized.min() >= 0
        assert quantized.max() < n_colors


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset_valid(self, temp_dir, sample_image):
        """Test loading dataset from directory structure."""
        # Create test dataset structure
        class1_dir = Path(temp_dir) / "class1"
        class2_dir = Path(temp_dir) / "class2"
        class1_dir.mkdir()
        class2_dir.mkdir()

        # Save test images
        cv2.imwrite(str(class1_dir / "img1.jpg"), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(class2_dir / "img2.jpg"), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))

        images, labels, paths = load_dataset(temp_dir)

        assert len(images) == 2
        assert len(labels) == 2
        assert len(paths) == 2
        assert set(labels) == {"class1", "class2"}

    def test_load_dataset_empty_directory(self, temp_dir):
        """Test loading from empty directory raises error."""
        with pytest.raises(ValueError, match="Khong tim thay thu muc con nao"):
            load_dataset(temp_dir)

    def test_load_dataset_custom_size(self, temp_dir, sample_image):
        """Test loading dataset with custom image size."""
        class_dir = Path(temp_dir) / "class1"
        class_dir.mkdir()
        cv2.imwrite(str(class_dir / "img1.jpg"), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))

        images, _, _ = load_dataset(temp_dir, size=(64, 64))

        assert images[0].shape == (64, 64, 3)


class TestGetClassNames:
    """Tests for get_class_names function."""

    def test_get_class_names_valid(self, temp_dir):
        """Test getting class names from directory."""
        # Create class directories
        (Path(temp_dir) / "class_a").mkdir()
        (Path(temp_dir) / "class_b").mkdir()
        (Path(temp_dir) / "class_c").mkdir()

        class_names = get_class_names(temp_dir)

        assert len(class_names) == 3
        assert class_names == ["class_a", "class_b", "class_c"]

    def test_get_class_names_sorted(self, temp_dir):
        """Test class names are sorted."""
        # Create class directories in non-alphabetical order
        (Path(temp_dir) / "zebra").mkdir()
        (Path(temp_dir) / "apple").mkdir()
        (Path(temp_dir) / "monkey").mkdir()

        class_names = get_class_names(temp_dir)

        assert class_names == sorted(class_names)

    def test_get_class_names_empty(self, temp_dir):
        """Test getting class names from empty directory."""
        class_names = get_class_names(temp_dir)
        assert class_names == []


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_image_invalid_size_type(self, sample_image_file):
        """Test load_image with invalid size type."""
        with pytest.raises(ValueError):
            load_image(sample_image_file, size="invalid")

    def test_load_image_invalid_size_values(self, sample_image_file):
        """Test load_image with invalid size values."""
        with pytest.raises(ValueError):
            load_image(sample_image_file, size=(0, 128))

    def test_load_image_negative_size(self, sample_image_file):
        """Test load_image with negative size."""
        with pytest.raises(ValueError):
            load_image(sample_image_file, size=(-128, 128))

    def test_convert_to_hsv_invalid_type(self):
        """Test convert_to_hsv with invalid input type."""
        with pytest.raises(TypeError):
            convert_to_hsv("not an array")

    def test_convert_to_hsv_invalid_shape(self):
        """Test convert_to_hsv with invalid shape."""
        with pytest.raises(ValueError):
            convert_to_hsv(np.zeros((100, 100), dtype=np.uint8))

    def test_convert_to_hsv_invalid_channels(self):
        """Test convert_to_hsv with invalid number of channels."""
        with pytest.raises(ValueError):
            convert_to_hsv(np.zeros((100, 100, 4), dtype=np.uint8))

    def test_quantize_colors_hsv_invalid_bins(self, sample_image):
        """Test quantize_colors_hsv with invalid bins."""
        img_hsv = convert_to_hsv(sample_image)
        with pytest.raises(ValueError):
            quantize_colors_hsv(img_hsv, h_bins=0)

    def test_quantize_colors_hsv_negative_bins(self, sample_image):
        """Test quantize_colors_hsv with negative bins."""
        img_hsv = convert_to_hsv(sample_image)
        with pytest.raises(ValueError):
            quantize_colors_hsv(img_hsv, s_bins=-1)

    def test_quantize_colors_rgb_invalid_bins(self, sample_image):
        """Test quantize_colors_rgb with invalid bins."""
        with pytest.raises(ValueError):
            quantize_colors_rgb(sample_image, bins=0)

    def test_quantize_colors_rgb_negative_bins(self, sample_image):
        """Test quantize_colors_rgb with negative bins."""
        with pytest.raises(ValueError):
            quantize_colors_rgb(sample_image, bins=-5)

    def test_load_dataset_nonexistent_directory(self):
        """Test load_dataset with nonexistent directory."""
        with pytest.raises(ValueError, match="khong ton tai"):
            load_dataset("/nonexistent/path/to/dataset")

    def test_load_dataset_file_not_directory(self, temp_dir, sample_image_file):
        """Test load_dataset with file instead of directory."""
        with pytest.raises(ValueError, match="khong phai la thu muc"):
            load_dataset(sample_image_file)

    def test_get_class_names_nonexistent_directory(self):
        """Test get_class_names with nonexistent directory."""
        with pytest.raises(ValueError, match="khong ton tai"):
            get_class_names("/nonexistent/path")

    def test_get_class_names_file_not_directory(self, sample_image_file):
        """Test get_class_names with file instead of directory."""
        with pytest.raises(ValueError, match="khong phai la thu muc"):
            get_class_names(sample_image_file)

    def test_quantize_colors_hsv_extreme_values(self, sample_image):
        """Test quantize_colors_hsv with extreme bin values."""
        img_hsv = convert_to_hsv(sample_image)
        quantized, n_colors = quantize_colors_hsv(img_hsv, h_bins=16, s_bins=8, v_bins=8)
        assert n_colors == 16 * 8 * 8
        assert np.all(quantized >= 0)
        assert np.all(quantized < n_colors)

    def test_quantize_colors_rgb_extreme_values(self, sample_image):
        """Test quantize_colors_rgb with extreme bin values."""
        quantized, n_colors = quantize_colors_rgb(sample_image, bins=16)
        assert n_colors == 16 ** 3
        assert np.all(quantized >= 0)
        assert np.all(quantized < n_colors)

    def test_load_image_file_io_error(self, temp_dir):
        """Test load_image handles file I/O errors gracefully."""
        # Create a file that can't be read as an image
        bad_file = Path(temp_dir) / "not_an_image.txt"
        bad_file.write_text("This is not an image")
        with pytest.raises(ValueError, match="Khong the doc"):
            load_image(str(bad_file))

    def test_convert_to_hsv_all_black(self):
        """Test convert_to_hsv with all black image."""
        black_img = np.zeros((50, 50, 3), dtype=np.uint8)
        hsv = convert_to_hsv(black_img)
        assert hsv is not None
        assert hsv.shape == (50, 50, 3)

    def test_convert_to_hsv_all_white(self):
        """Test convert_to_hsv with all white image."""
        white_img = np.full((50, 50, 3), 255, dtype=np.uint8)
        hsv = convert_to_hsv(white_img)
        assert hsv is not None
        assert hsv.shape == (50, 50, 3)

    def test_quantize_colors_hsv_all_same_color(self):
        """Test quantize_colors_hsv with uniform color image."""
        uniform_img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img_hsv = convert_to_hsv(uniform_img)
        quantized, n_colors = quantize_colors_hsv(img_hsv)
        assert quantized is not None
        # All pixels should map to same color
        assert len(np.unique(quantized)) <= 2

    def test_load_dataset_mixed_formats(self, temp_dir, sample_image):
        """Test load_dataset with mixed image formats."""
        class_dir = Path(temp_dir) / "mixed"
        class_dir.mkdir()

        # Save images in different formats
        img_bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(class_dir / "img1.jpg"), img_bgr)
        cv2.imwrite(str(class_dir / "img2.png"), img_bgr)

        images, labels, paths = load_dataset(temp_dir)
        assert len(images) == 2
        assert all(img.shape == (128, 128, 3) for img in images)
