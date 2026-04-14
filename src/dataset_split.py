"""
dataset_split.py - Tao va tai split train/test co dinh cho Corel-1K.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from dataset_profile import (
    DEFAULT_DATASET_PROFILE,
    get_profile_config,
    get_split_filename,
)


DEFAULT_SPLIT_FILENAME = get_split_filename(DEFAULT_DATASET_PROFILE)
DEFAULT_DATASET_NAME = get_profile_config(DEFAULT_DATASET_PROFILE)["dataset_name"]
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.0
DEFAULT_TEST_RATIO = 0.2


def _normalize_relative_path(image_path, data_dir):
    """Chuyen duong dan anh thanh path tuong doi theo data_dir."""
    return Path(image_path).resolve().relative_to(Path(data_dir).resolve()).as_posix()


def _class_distribution(label_names):
    counts = Counter(label_names)
    return {str(label): int(count) for label, count in sorted(counts.items())}


def build_split_metadata(image_paths, label_names, data_dir,
                         train_ratio=DEFAULT_TRAIN_RATIO,
                         val_ratio=DEFAULT_VAL_RATIO,
                         test_ratio=DEFAULT_TEST_RATIO,
                         random_state=DEFAULT_RANDOM_STATE,
                         dataset_name=DEFAULT_DATASET_NAME):
    """Tao metadata split stratified co dinh."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("Tong ti le train/test phai bang 1.0")

    indices = np.arange(len(image_paths))
    label_names = np.asarray(label_names)

    train_idx, test_idx, train_labels, test_labels = train_test_split(
        indices,
        label_names,
        test_size=test_ratio,
        random_state=random_state,
        stratify=label_names,
        shuffle=True,
    )

    split_indices = {
        'train': np.sort(train_idx),
        'test': np.sort(test_idx),
    }

    metadata = {
        'dataset_name': str(dataset_name),
        'data_dir': Path(data_dir).as_posix(),
        'random_state': int(random_state),
        'ratios': {
            'train': float(train_ratio),
            'test': float(test_ratio),
        },
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'path_mode': 'relative_to_data_dir',
        'splits': {},
        'counts': {},
        'class_distribution': {},
    }

    for split_name, split_idx in split_indices.items():
        rel_paths = [_normalize_relative_path(image_paths[i], data_dir) for i in split_idx]
        split_label_names = [str(label_names[i]) for i in split_idx]
        metadata['splits'][split_name] = rel_paths
        metadata['counts'][split_name] = int(len(split_idx))
        metadata['class_distribution'][split_name] = _class_distribution(split_label_names)

    return metadata


def save_split_metadata(split_path, metadata):
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_split_metadata(split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_and_save_split(split_path, image_paths, label_names, data_dir,
                          train_ratio=DEFAULT_TRAIN_RATIO,
                          val_ratio=DEFAULT_VAL_RATIO,
                          test_ratio=DEFAULT_TEST_RATIO,
                          random_state=DEFAULT_RANDOM_STATE,
                          dataset_name=DEFAULT_DATASET_NAME):
    metadata = build_split_metadata(
        image_paths=image_paths,
        label_names=label_names,
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        dataset_name=dataset_name,
    )
    save_split_metadata(split_path, metadata)
    return metadata


def ensure_split_metadata(split_path, image_paths, label_names, data_dir,
                          train_ratio=DEFAULT_TRAIN_RATIO,
                          val_ratio=DEFAULT_VAL_RATIO,
                          test_ratio=DEFAULT_TEST_RATIO,
                          random_state=DEFAULT_RANDOM_STATE,
                          dataset_name=DEFAULT_DATASET_NAME,
                          force=False):
    """Tai split metadata, neu chua co thi tao moi."""
    split_path = Path(split_path)
    if force or not split_path.exists():
        return create_and_save_split(
            split_path=split_path,
            image_paths=image_paths,
            label_names=label_names,
            data_dir=data_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            dataset_name=dataset_name,
        )
    return load_split_metadata(split_path)


def validate_split_metadata(split_metadata, expected_dataset_name=None):
    """Xac thuc split metadata co thuoc dung dataset hay khong."""
    if expected_dataset_name is None:
        return

    actual_name = split_metadata.get('dataset_name')
    if actual_name and str(actual_name) != str(expected_dataset_name):
        raise ValueError(
            f"Split metadata thuoc dataset '{actual_name}', "
            f"khong trung voi dataset dang chon '{expected_dataset_name}'."
        )


def resolve_split_indices(image_paths, data_dir, split_metadata):
    """Map split metadata ve index theo thu tu image_paths hien tai."""
    image_paths = np.asarray(image_paths)
    rel_to_index = {
        _normalize_relative_path(path, data_dir): idx
        for idx, path in enumerate(image_paths)
    }

    split_indices = {}
    for split_name, rel_paths in split_metadata['splits'].items():
        indices = []
        missing_paths = []
        for rel_path in rel_paths:
            idx = rel_to_index.get(rel_path)
            if idx is None:
                missing_paths.append(rel_path)
            else:
                indices.append(idx)
        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            raise FileNotFoundError(
                f"Khong map duoc {len(missing_paths)} anh trong split '{split_name}' "
                f"ve image_paths hien tai. Vi du: {preview}"
            )
        split_indices[split_name] = np.array(sorted(indices), dtype=np.int32)

    return split_indices


def merge_split_indices(split_indices, split_names):
    """Noi nhieu split thanh 1 mang index da sort."""
    merged = np.concatenate([split_indices[name] for name in split_names])
    return np.array(sorted(merged.tolist()), dtype=np.int32)
