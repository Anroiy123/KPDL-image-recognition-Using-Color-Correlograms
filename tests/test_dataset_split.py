"""Unit tests for dataset_split.py module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.dataset_split import (
    build_split_metadata,
    create_and_save_split,
    ensure_split_metadata,
    load_split_metadata,
    merge_split_indices,
    resolve_split_indices,
    save_split_metadata,
    validate_split_metadata,
)


class TestBuildSplitMetadata:
    """Tests for build_split_metadata function."""

    def test_build_split_metadata_basic(self, temp_dir):
        """Test building split metadata."""
        # Create enough samples for stratified split (at least 2 per class for test set)
        image_paths = [f"{temp_dir}/class{i%2}/img{i}.jpg" for i in range(20)]
        label_names = [f"class{i%2}" for i in range(20)]

        metadata = build_split_metadata(
            image_paths, label_names, temp_dir, train_ratio=0.8, test_ratio=0.2
        )

        assert metadata is not None
        assert "splits" in metadata
        assert "train" in metadata["splits"]
        assert "test" in metadata["splits"]
        assert metadata["ratios"]["train"] == 0.8
        assert metadata["ratios"]["test"] == 0.2

    def test_build_split_metadata_stratified(self, temp_dir):
        """Test split is stratified."""
        image_paths = [f"{temp_dir}/class{i%2}/img{i}.jpg" for i in range(100)]
        label_names = [f"class{i%2}" for i in range(100)]

        metadata = build_split_metadata(
            image_paths, label_names, temp_dir, train_ratio=0.8, test_ratio=0.2
        )

        # Check class distribution is preserved
        train_dist = metadata["class_distribution"]["train"]
        test_dist = metadata["class_distribution"]["test"]

        assert "class0" in train_dist
        assert "class1" in train_dist
        assert "class0" in test_dist
        assert "class1" in test_dist

    def test_build_split_metadata_invalid_ratios(self, temp_dir):
        """Test invalid ratios raise error."""
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(10)]
        label_names = ["class1"] * 10

        with pytest.raises(ValueError, match="Tong ti le"):
            build_split_metadata(
                image_paths, label_names, temp_dir, train_ratio=0.5, test_ratio=0.3
            )

    def test_build_split_metadata_counts(self, temp_dir):
        """Test split counts are correct."""
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(100)]
        label_names = ["class1"] * 100

        metadata = build_split_metadata(
            image_paths, label_names, temp_dir, train_ratio=0.8, test_ratio=0.2
        )

        assert metadata["counts"]["train"] == 80
        assert metadata["counts"]["test"] == 20


class TestSaveSplitMetadata:
    """Tests for save_split_metadata function."""

    def test_save_split_metadata_creates_file(self, temp_dir):
        """Test saving split metadata creates file."""
        split_path = Path(temp_dir) / "split.json"
        metadata = {
            "dataset_name": "test",
            "splits": {"train": [], "test": []},
            "counts": {"train": 0, "test": 0},
        }

        save_split_metadata(split_path, metadata)

        assert split_path.exists()

    def test_save_split_metadata_content(self, temp_dir):
        """Test saved metadata content is correct."""
        split_path = Path(temp_dir) / "split.json"
        metadata = {
            "dataset_name": "test",
            "splits": {"train": ["img1.jpg"], "test": ["img2.jpg"]},
        }

        save_split_metadata(split_path, metadata)

        with open(split_path) as f:
            loaded = json.load(f)

        assert loaded["dataset_name"] == "test"
        assert loaded["splits"]["train"] == ["img1.jpg"]


class TestLoadSplitMetadata:
    """Tests for load_split_metadata function."""

    def test_load_split_metadata_valid(self, temp_dir):
        """Test loading valid split metadata."""
        split_path = Path(temp_dir) / "split.json"
        metadata = {
            "dataset_name": "test",
            "splits": {"train": ["img1.jpg"], "test": ["img2.jpg"]},
        }

        save_split_metadata(split_path, metadata)
        loaded = load_split_metadata(split_path)

        assert loaded["dataset_name"] == "test"
        assert loaded["splits"]["train"] == ["img1.jpg"]


class TestValidateSplitMetadata:
    """Tests for validate_split_metadata function."""

    def test_validate_split_metadata_matching(self):
        """Test validation passes for matching dataset."""
        metadata = {"dataset_name": "corel-1k"}

        # Should not raise
        validate_split_metadata(metadata, expected_dataset_name="corel-1k")

    def test_validate_split_metadata_mismatch(self):
        """Test validation fails for mismatched dataset."""
        metadata = {"dataset_name": "corel-1k"}

        with pytest.raises(ValueError, match="thuoc dataset"):
            validate_split_metadata(metadata, expected_dataset_name="other-dataset")

    def test_validate_split_metadata_none_expected(self):
        """Test validation skips when expected is None."""
        metadata = {"dataset_name": "corel-1k"}

        # Should not raise
        validate_split_metadata(metadata, expected_dataset_name=None)


class TestResolveSplitIndices:
    """Tests for resolve_split_indices function."""

    def test_resolve_split_indices_basic(self, temp_dir):
        """Test resolving split indices."""
        image_paths = [
            f"{temp_dir}/class1/img1.jpg",
            f"{temp_dir}/class1/img2.jpg",
            f"{temp_dir}/class2/img3.jpg",
        ]
        split_metadata = {
            "splits": {
                "train": ["class1/img1.jpg", "class2/img3.jpg"],
                "test": ["class1/img2.jpg"],
            }
        }

        indices = resolve_split_indices(image_paths, temp_dir, split_metadata)

        assert "train" in indices
        assert "test" in indices
        assert len(indices["train"]) == 2
        assert len(indices["test"]) == 1

    def test_resolve_split_indices_missing_paths(self, temp_dir):
        """Test missing paths raise error."""
        image_paths = [f"{temp_dir}/class1/img1.jpg"]
        split_metadata = {
            "splits": {
                "train": ["class1/img1.jpg", "class1/missing.jpg"],
            }
        }

        with pytest.raises(FileNotFoundError, match="Khong map duoc"):
            resolve_split_indices(image_paths, temp_dir, split_metadata)


class TestMergeSplitIndices:
    """Tests for merge_split_indices function."""

    def test_merge_split_indices_basic(self):
        """Test merging split indices."""
        split_indices = {
            "train": np.array([0, 2, 4]),
            "test": np.array([1, 3]),
        }

        merged = merge_split_indices(split_indices, ["train", "test"])

        assert len(merged) == 5
        assert np.array_equal(merged, np.array([0, 1, 2, 3, 4]))

    def test_merge_split_indices_sorted(self):
        """Test merged indices are sorted."""
        split_indices = {
            "train": np.array([4, 2, 0]),
            "test": np.array([3, 1]),
        }

        merged = merge_split_indices(split_indices, ["train", "test"])

        assert np.array_equal(merged, np.array([0, 1, 2, 3, 4]))


class TestCreateAndSaveSplit:
    """Tests for create_and_save_split function."""

    def test_create_and_save_split(self, temp_dir):
        """Test creating and saving split."""
        split_path = Path(temp_dir) / "split.json"
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(10)]
        label_names = ["class1"] * 10

        metadata = create_and_save_split(
            split_path, image_paths, label_names, temp_dir
        )

        assert split_path.exists()
        assert metadata["counts"]["train"] == 8
        assert metadata["counts"]["test"] == 2


class TestEnsureSplitMetadata:
    """Tests for ensure_split_metadata function."""

    def test_ensure_split_metadata_creates_if_missing(self, temp_dir):
        """Test creating split if missing."""
        split_path = Path(temp_dir) / "split.json"
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(10)]
        label_names = ["class1"] * 10

        metadata = ensure_split_metadata(
            split_path, image_paths, label_names, temp_dir
        )

        assert split_path.exists()
        assert metadata is not None

    def test_ensure_split_metadata_loads_existing(self, temp_dir):
        """Test loading existing split."""
        split_path = Path(temp_dir) / "split.json"
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(10)]
        label_names = ["class1"] * 10

        # Create first
        metadata1 = ensure_split_metadata(
            split_path, image_paths, label_names, temp_dir
        )

        # Load second
        metadata2 = ensure_split_metadata(
            split_path, image_paths, label_names, temp_dir
        )

        assert metadata1["random_state"] == metadata2["random_state"]

    def test_ensure_split_metadata_force_recreate(self, temp_dir):
        """Test force recreating split."""
        split_path = Path(temp_dir) / "split.json"
        image_paths = [f"{temp_dir}/class1/img{i}.jpg" for i in range(10)]
        label_names = ["class1"] * 10

        metadata1 = ensure_split_metadata(
            split_path, image_paths, label_names, temp_dir, random_state=42
        )

        metadata2 = ensure_split_metadata(
            split_path, image_paths, label_names, temp_dir, random_state=43, force=True
        )

        assert metadata1["random_state"] != metadata2["random_state"]
