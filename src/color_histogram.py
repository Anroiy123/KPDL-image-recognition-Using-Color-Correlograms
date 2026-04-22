"""
color_histogram.py - Cai dat Color Histogram (baseline de so sanh voi Correlogram)

Color Histogram chi dem tan suat xuat hien cua moi mau trong anh.
KHONG co thong tin khong gian (spatial information).
Dung de chung minh Color Correlogram tot hon.
"""

from typing import Tuple
import numpy as np
import cv2


def _validate_quantized_image(
    quantized_img: np.ndarray, param_name: str = "quantized_img"
) -> None:
    """Validate quantized image array."""
    if not isinstance(quantized_img, np.ndarray):
        raise TypeError(f"{param_name} must be numpy array, got {type(quantized_img)}")
    if quantized_img.ndim != 2:
        raise ValueError(
            f"{param_name} must be 2D array, got shape {quantized_img.shape}"
        )


def _validate_n_colors(n_colors: int, param_name: str = "n_colors") -> None:
    """Validate n_colors parameter."""
    if not isinstance(n_colors, (int, np.integer)) or n_colors <= 0:
        raise ValueError(f"{param_name} must be positive integer, got {n_colors}")


def color_histogram(quantized_img: np.ndarray, n_colors: int) -> np.ndarray:
    """Tinh Color Histogram cho anh da luong tu hoa.

    Chi don gian dem so pixel cua moi mau va chuan hoa.

    Args:
        quantized_img: Ma tran 2D (H x W), moi pixel la ma mau
        n_colors: Tong so mau sau luong tu hoa

    Returns:
        histogram: Vector tan suat, kich thuoc (n_colors,), tong = 1.0

    Raises:
        TypeError: If quantized_img is not a valid array
        ValueError: If n_colors is invalid
    """
    _validate_quantized_image(quantized_img)
    _validate_n_colors(n_colors)

    histogram = np.bincount(quantized_img.ravel(), minlength=n_colors).astype(
        np.float64
    )

    # Chuan hoa: chia cho tong so pixel
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    return histogram


def extract_histogram_feature(
    img_bgr: np.ndarray,
    color_space: str = "hsv",
    h_bins: int = 8,
    s_bins: int = 3,
    v_bins: int = 3,
    rgb_bins: int = 4,
) -> np.ndarray:
    """Ham tien ich: tu anh BGR goc -> vector histogram.

    Args:
        img_bgr: Anh BGR (numpy array)
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin cho HSV
        rgb_bins: So bin cho RGB

    Returns:
        feature: Vector histogram da chuan hoa

    Raises:
        ValueError: If color_space is invalid
    """
    if color_space == "hsv":
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        n_colors = h_bins * s_bins * v_bins

        h = img_hsv[:, :, 0].astype(np.int32)
        s = img_hsv[:, :, 1].astype(np.int32)
        v = img_hsv[:, :, 2].astype(np.int32)
        h_q = np.clip(h * h_bins // 180, 0, h_bins - 1).astype(np.int32)
        s_q = np.clip(s * s_bins // 256, 0, s_bins - 1).astype(np.int32)
        v_q = np.clip(v * v_bins // 256, 0, v_bins - 1).astype(np.int32)
        quantized = h_q * (s_bins * v_bins) + s_q * v_bins + v_q

    elif color_space == "rgb":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        n_colors = rgb_bins**3

        r = np.clip(
            img_rgb[:, :, 0].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1
        ).astype(np.int32)
        g = np.clip(
            img_rgb[:, :, 1].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1
        ).astype(np.int32)
        b = np.clip(
            img_rgb[:, :, 2].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1
        ).astype(np.int32)
        quantized = r * (rgb_bins * rgb_bins) + g * rgb_bins + b
    else:
        raise ValueError(
            f"color_space phai la 'hsv' hoac 'rgb', khong phai '{color_space}'"
        )

    return color_histogram(quantized, n_colors)


if __name__ == "__main__":
    import sys

    # Test nhanh
    print("Test Color Histogram...")
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    feat_hsv = extract_histogram_feature(img, color_space="hsv")
    feat_rgb = extract_histogram_feature(img, color_space="rgb")

    print(f"HSV Histogram: size={feat_hsv.shape}, sum={feat_hsv.sum():.4f}")
    print(f"RGB Histogram: size={feat_rgb.shape}, sum={feat_rgb.sum():.4f}")
    print(f"HSV min={feat_hsv.min():.6f}, max={feat_hsv.max():.6f}")


if __name__ == "__main__":
    import sys

    # Test nhanh
    print("Test Color Histogram...")
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    feat_hsv = extract_histogram_feature(img, color_space="hsv")
    feat_rgb = extract_histogram_feature(img, color_space="rgb")

    print(f"HSV Histogram: size={feat_hsv.shape}, sum={feat_hsv.sum():.4f}")
    print(f"RGB Histogram: size={feat_rgb.shape}, sum={feat_rgb.sum():.4f}")
    print(f"HSV min={feat_hsv.min():.6f}, max={feat_hsv.max():.6f}")
