"""
feature_extraction.py - Trich xuat dac trung cho toan bo dataset

Trich xuat Color Correlogram va Color Histogram cho tat ca anh,
luu ket qua vao file .npy de su dung lai.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Them duong dan src vao sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_dataset, convert_to_hsv, quantize_colors_hsv, quantize_colors_rgb
from color_correlogram import auto_correlogram_fast
from color_histogram import color_histogram


def extract_all_features(images, method='correlogram', color_space='rgb',
                          h_bins=8, s_bins=3, v_bins=3, rgb_bins=4,
                          distances=None):
    """Trich xuat dac trung cho danh sach anh.

    Args:
        images: List cac anh BGR
        method: 'correlogram' hoac 'histogram'
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin HSV
        rgb_bins: So bin RGB
        distances: List khoang cach cho correlogram

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
            feat = auto_correlogram_fast(quantized, n_colors, distances)
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


def main():
    """Trich xuat tat ca dac trung va luu vao file."""

    # Xac dinh duong dan
    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = project_dir / "data" / "corel-1k"
    features_dir = project_dir / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRICH XUAT DAC TRUNG - COLOR CORRELOGRAM PROJECT")
    print("=" * 60)

    # Tai dataset
    print("\n[1/5] Tai dataset...")
    images, labels, paths = load_dataset(data_dir)

    if len(images) == 0:
        print("ERROR: Khong co anh nao! Hay kiem tra thu muc data/corel-1k/")
        return

    # Chuyen labels thanh so
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    print(f"\nCac lop: {list(class_names)}")
    print(f"So chieu label: {y.shape}")

    # Trich xuat Correlogram HSV
    print("\n[2/5] Trich xuat Color Correlogram (HSV)...")
    start = time.time()
    X_corr_hsv = extract_all_features(images, method='correlogram', color_space='hsv')
    t_corr_hsv = time.time() - start
    print(f"  Xong! {t_corr_hsv:.1f}s, shape: {X_corr_hsv.shape}")

    # Trich xuat Correlogram RGB
    print("\n[3/5] Trich xuat Color Correlogram (RGB)...")
    start = time.time()
    X_corr_rgb = extract_all_features(images, method='correlogram', color_space='rgb')
    t_corr_rgb = time.time() - start
    print(f"  Xong! {t_corr_rgb:.1f}s, shape: {X_corr_rgb.shape}")

    # Trich xuat Histogram HSV
    print("\n[4/6] Trich xuat Color Histogram (HSV)...")
    start = time.time()
    X_hist_hsv = extract_all_features(images, method='histogram', color_space='hsv')
    t_hist_hsv = time.time() - start
    print(f"  Xong! {t_hist_hsv:.1f}s, shape: {X_hist_hsv.shape}")

    # Trich xuat Histogram RGB
    print("\n[5/6] Trich xuat Color Histogram (RGB)...")
    start = time.time()
    X_hist_rgb = extract_all_features(images, method='histogram', color_space='rgb')
    t_hist_rgb = time.time() - start
    print(f"  Xong! {t_hist_rgb:.1f}s, shape: {X_hist_rgb.shape}")

    # Luu tat ca
    print("\n[6/6] Luu ket qua...")
    np.save(features_dir / "correlogram_hsv.npy", X_corr_hsv)
    np.save(features_dir / "correlogram_rgb.npy", X_corr_rgb)
    np.save(features_dir / "histogram_hsv.npy", X_hist_hsv)
    np.save(features_dir / "histogram_rgb.npy", X_hist_rgb)
    np.save(features_dir / "labels.npy", y)
    np.save(features_dir / "class_names.npy", class_names)
    np.save(features_dir / "image_paths.npy", np.array(paths))

    print(f"\nDa luu vao: {features_dir}")
    print(f"  - correlogram_hsv.npy: {X_corr_hsv.shape}")
    print(f"  - correlogram_rgb.npy: {X_corr_rgb.shape}")
    print(f"  - histogram_hsv.npy:   {X_hist_hsv.shape}")
    print(f"  - histogram_rgb.npy:   {X_hist_rgb.shape}")
    print(f"  - labels.npy:          {y.shape}")
    print(f"  - class_names.npy:     {class_names.shape}")

    print("\n" + "=" * 60)
    print("TRICH XUAT HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
