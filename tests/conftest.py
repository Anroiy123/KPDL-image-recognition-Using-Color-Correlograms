"""Pytest configuration and fixtures for thesis project tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with random colors
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_image_file(temp_dir: str, sample_image: np.ndarray) -> str:
    """Create a temporary image file for testing."""
    image_path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    return image_path


@pytest.fixture
def sample_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Create a sample dataset with features and labels."""
    # Create 100 samples with 256 features each, with balanced classes
    X = np.random.rand(100, 256).astype(np.float32)
    # Create labels for 10 classes with 10 samples each
    y = np.repeat(np.arange(10), 10)
    return X, y


@pytest.fixture
def sample_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample predictions and ground truth labels."""
    # Create 100 predictions and labels
    y_true = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    return y_true, y_pred


@pytest.fixture
def sample_correlogram() -> np.ndarray:
    """Create a sample color correlogram feature vector."""
    # Color correlogram typically has 256 dimensions (for 256 color levels)
    correlogram = np.random.rand(256).astype(np.float32)
    # Normalize to sum to 1
    correlogram = correlogram / correlogram.sum()
    return correlogram


@pytest.fixture
def sample_histogram() -> np.ndarray:
    """Create a sample color histogram feature vector."""
    # Color histogram typically has 256 dimensions
    histogram = np.random.rand(256).astype(np.float32)
    # Normalize to sum to 1
    histogram = histogram / histogram.sum()
    return histogram
