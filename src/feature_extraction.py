"""
feature_extraction.py - Trich xuat dac trung cho toan bo dataset

Trich xuat Color Correlogram va Color Histogram cho tat ca anh,
luu ket qua vao file .npy de su dung lai.
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Them duong dan src vao sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_dataset, convert_to_hsv, quantize_colors_hsv, quantize_colors_rgb
from color_correlogram import auto_correlogram_fast, spatial_correlogram
from color_histogram import color_histogram
from dataset_split import ensure_split_metadata
from dataset_profile import (
    DEFAULT_DATASET_PROFILE,
    list_dataset_profiles,
    resolve_dataset_profile,
    scoped_artifact_path,
)


FEATURE_OUTPUT_FILES = {
    "correlogram_hsv": "correlogram_hsv.npy",
    "correlogram_hsv_spatial": "correlogram_hsv_spatial.npy",
    "correlogram_rgb": "correlogram_rgb.npy",
    "histogram_hsv": "histogram_hsv.npy",
    "histogram_rgb": "histogram_rgb.npy",
    "labels": "labels.npy",
    "class_names": "class_names.npy",
    "image_paths": "image_paths.npy",
}


def extract_all_features(images, method='correlogram', color_space='rgb',
                         h_bins=8, s_bins=3, v_bins=3, rgb_bins=4,
                         distances=None, spatial_grid=None, include_global=True):
    """Trich xuat dac trung cho danh sach anh.

    Args:
        images: List cac anh BGR
        method: 'correlogram' hoac 'histogram'
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin HSV
        rgb_bins: So bin RGB
        distances: List khoang cach cho correlogram
        spatial_grid: Neu khac None, noi them correlogram cho tung o trong luoi
        include_global: Co giu correlogram toan anh khi dung spatial_grid

    Returns:
        features: Ma tran (N x D), N anh, D chieu dac trung
    """
    import cv2

    if distances is None:
        distances = [1, 3, 5, 7]

    features = []
    total = len(images)

    for i, img in enumerate(images):
        # Luong tu hoa
        if color_space == 'hsv':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            quantized, n_colors = quantize_colors_hsv(img_hsv, h_bins, s_bins, v_bins)
        else:
            quantized, n_colors = quantize_colors_rgb(img, rgb_bins)

        # Trich xuat dac trung
        if method == 'correlogram':
            if spatial_grid is None:
                feat = auto_correlogram_fast(quantized, n_colors, distances)
            else:
                feat = spatial_correlogram(
                    quantized,
                    n_colors,
                    distances=distances,
                    grid_size=spatial_grid,
                    include_global=include_global,
                )
        elif method == 'histogram':
            feat = color_histogram(quantized, n_colors)
        else:
            raise ValueError(f"method phai la 'correlogram' hoac 'histogram'")

        features.append(feat)

        # Hien thi tien do
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{method}/{color_space}] {i+1}/{total} "
                  f"({100*(i+1)/total:.0f}%)")

    return np.array(features)


def parse_args():
    parser = argparse.ArgumentParser(description="Trich xuat dac trung cho dataset profile")
    parser.add_argument(
        "--dataset-profile",
        default=DEFAULT_DATASET_PROFILE,
        choices=list_dataset_profiles(),
        help="Profile dataset can trich xuat (mac dinh: corel-1k)",
    )
    return parser.parse_args()


def main(dataset_profile_key=DEFAULT_DATASET_PROFILE):
    """Trich xuat tat ca dac trung va luu vao file."""

    # Xac dinh duong dan
    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_profile = resolve_dataset_profile(dataset_profile_key, project_dir)
    data_dir = dataset_profile["data_dir"]
    features_dir = project_dir / "data" / "features"
    splits_dir = project_dir / "data" / "splits"
    features_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRICH XUAT DAC TRUNG - COLOR CORRELOGRAM PROJECT")
    print("=" * 60)
    print(f"Dataset profile: {dataset_profile['key']} ({dataset_profile['display_name']})")
    print(f"Data dir: {data_dir}")

    # Tai dataset
    print("\n[1/7] Tai dataset...")
    images, labels, paths = load_dataset(data_dir)

    if len(images) == 0:
        print(f"ERROR: Khong co anh nao! Hay kiem tra thu muc {data_dir}")
        return

    # Chuyen labels thanh so
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    print(f"\nCac lop: {list(class_names)}")
    print(f"So chieu label: {y.shape}")

    # Trich xuat Correlogram HSV
    print("\n[2/7] Trich xuat Color Correlogram (HSV)...")
    start = time.time()
    X_corr_hsv = extract_all_features(images, method='correlogram', color_space='hsv')
    t_corr_hsv = time.time() - start
    print(f"  Xong! {t_corr_hsv:.1f}s, shape: {X_corr_hsv.shape}")

    # Trich xuat Spatial Correlogram HSV
    print("\n[3/7] Trich xuat Spatial Color Correlogram (HSV, global + 2x2)...")
    start = time.time()
    X_corr_hsv_spatial = extract_all_features(
        images,
        method='correlogram',
        color_space='hsv',
        spatial_grid=2,
        include_global=True,
    )
    t_corr_hsv_spatial = time.time() - start
    print(f"  Xong! {t_corr_hsv_spatial:.1f}s, shape: {X_corr_hsv_spatial.shape}")

    # Trich xuat Correlogram RGB
    print("\n[4/7] Trich xuat Color Correlogram (RGB)...")
    start = time.time()
    X_corr_rgb = extract_all_features(images, method='correlogram', color_space='rgb')
    t_corr_rgb = time.time() - start
    print(f"  Xong! {t_corr_rgb:.1f}s, shape: {X_corr_rgb.shape}")

    # Trich xuat Histogram HSV
    print("\n[5/7] Trich xuat Color Histogram (HSV)...")
    start = time.time()
    X_hist_hsv = extract_all_features(images, method='histogram', color_space='hsv')
    t_hist_hsv = time.time() - start
    print(f"  Xong! {t_hist_hsv:.1f}s, shape: {X_hist_hsv.shape}")

    # Trich xuat Histogram RGB
    print("\n[6/7] Trich xuat Color Histogram (RGB)...")
    start = time.time()
    X_hist_rgb = extract_all_features(images, method='histogram', color_space='rgb')
    t_hist_rgb = time.time() - start
    print(f"  Xong! {t_hist_rgb:.1f}s, shape: {X_hist_rgb.shape}")

    # Luu tat ca
    print("\n[7/7] Luu ket qua...")
    output_values = {
        "correlogram_hsv": X_corr_hsv,
        "correlogram_hsv_spatial": X_corr_hsv_spatial,
        "correlogram_rgb": X_corr_rgb,
        "histogram_hsv": X_hist_hsv,
        "histogram_rgb": X_hist_rgb,
        "labels": y,
        "class_names": class_names,
        "image_paths": np.array(paths),
    }
    output_paths = {}
    for key, value in output_values.items():
        output_path = scoped_artifact_path(features_dir, dataset_profile["key"], FEATURE_OUTPUT_FILES[key])
        np.save(output_path, value)
        output_paths[key] = output_path

    split_path = splits_dir / dataset_profile["split_filename"]
    split_metadata = ensure_split_metadata(
        split_path=split_path,
        image_paths=paths,
        label_names=labels,
        data_dir=data_dir,
        dataset_name=dataset_profile["dataset_name"],
        force=True,
    )

    print(f"\nDa luu vao: {features_dir}")
    print(f"  - {output_paths['correlogram_hsv'].name}: {X_corr_hsv.shape}")
    print(f"  - {output_paths['correlogram_hsv_spatial'].name}: {X_corr_hsv_spatial.shape}")
    print(f"  - {output_paths['correlogram_rgb'].name}: {X_corr_rgb.shape}")
    print(f"  - {output_paths['histogram_hsv'].name}:   {X_hist_hsv.shape}")
    print(f"  - {output_paths['histogram_rgb'].name}:   {X_hist_rgb.shape}")
    print(f"  - {output_paths['labels'].name}:          {y.shape}")
    print(f"  - {output_paths['class_names'].name}:     {class_names.shape}")
    print(f"  - split metadata:      {split_path}")
    for split_name, count in split_metadata['counts'].items():
        print(f"      {split_name}: {count} anh")

    print("\n" + "=" * 60)
    print("TRICH XUAT HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(dataset_profile_key=args.dataset_profile)
