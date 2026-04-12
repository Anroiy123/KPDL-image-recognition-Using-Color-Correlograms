"""
color_histogram.py - Cai dat Color Histogram (baseline de so sanh voi Correlogram)

Color Histogram chi dem tan suat xuat hien cua moi mau trong anh.
KHONG co thong tin khong gian (spatial information).
Dung de chung minh Color Correlogram tot hon.
"""

import numpy as np
import cv2


def color_histogram(quantized_img, n_colors):
    """Tinh Color Histogram cho anh da luong tu hoa.

    Chi don gian dem so pixel cua moi mau va chuan hoa.

    Args:
        quantized_img: Ma tran 2D (H x W), moi pixel la ma mau
        n_colors: Tong so mau sau luong tu hoa

    Returns:
        histogram: Vector tan suat, kich thuoc (n_colors,), tong = 1.0
    """
    histogram = np.bincount(quantized_img.ravel(), minlength=n_colors).astype(np.float64)

    # Chuan hoa: chia cho tong so pixel
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    return histogram


def extract_histogram_feature(img_bgr, color_space='hsv',
                               h_bins=8, s_bins=3, v_bins=3,
                               rgb_bins=4):
    """Ham tien ich: tu anh BGR goc -> vector histogram.

    Args:
        img_bgr: Anh BGR (numpy array)
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin cho HSV
        rgb_bins: So bin cho RGB

    Returns:
        feature: Vector histogram da chuan hoa
    """
    if color_space == 'hsv':
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        n_colors = h_bins * s_bins * v_bins

        h = img_hsv[:, :, 0].astype(np.int32)
        s = img_hsv[:, :, 1].astype(np.int32)
        v = img_hsv[:, :, 2].astype(np.int32)
        h_q = np.clip(h * h_bins // 180, 0, h_bins - 1).astype(np.int32)
        s_q = np.clip(s * s_bins // 256, 0, s_bins - 1).astype(np.int32)
        v_q = np.clip(v * v_bins // 256, 0, v_bins - 1).astype(np.int32)
        quantized = h_q * (s_bins * v_bins) + s_q * v_bins + v_q

    elif color_space == 'rgb':
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        n_colors = rgb_bins ** 3

        r = np.clip(img_rgb[:, :, 0].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        g = np.clip(img_rgb[:, :, 1].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        b = np.clip(img_rgb[:, :, 2].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        quantized = r * (rgb_bins * rgb_bins) + g * rgb_bins + b
    else:
        raise ValueError(f"color_space phai la 'hsv' hoac 'rgb'")

    return color_histogram(quantized, n_colors)


if __name__ == "__main__":
    import sys

    # Test nhanh
    print("Test Color Histogram...")
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    feat_hsv = extract_histogram_feature(img, color_space='hsv')
    feat_rgb = extract_histogram_feature(img, color_space='rgb')

    print(f"HSV Histogram: size={feat_hsv.shape}, sum={feat_hsv.sum():.4f}")
    print(f"RGB Histogram: size={feat_rgb.shape}, sum={feat_rgb.sum():.4f}")
    print(f"HSV min={feat_hsv.min():.6f}, max={feat_hsv.max():.6f}")
