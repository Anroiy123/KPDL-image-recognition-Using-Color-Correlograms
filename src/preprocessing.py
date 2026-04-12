"""
preprocessing.py - Tien xu ly anh cho bai toan Color Correlogram
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_image(path, size=(128, 128)):
    """Doc anh va resize ve kich thuoc dong nhat.

    Args:
        path: Duong dan den file anh
        size: Kich thuoc dau ra (width, height)

    Returns:
        img: Anh BGR (numpy array) da resize
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Khong the doc anh: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def convert_to_hsv(img):
    """Chuyen anh tu BGR sang HSV.

    Args:
        img: Anh BGR (numpy array)

    Returns:
        img_hsv: Anh HSV (numpy array)
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def quantize_colors_hsv(img_hsv, h_bins=8, s_bins=3, v_bins=3):
    """Luong tu hoa mau sac trong khong gian HSV.

    Giam so mau tu hang trieu xuong con h_bins * s_bins * v_bins mau.

    Args:
        img_hsv: Anh HSV (numpy array)
        h_bins: So bin cho kenh Hue (0-179 trong OpenCV)
        s_bins: So bin cho kenh Saturation (0-255)
        v_bins: So bin cho kenh Value (0-255)

    Returns:
        quantized: Ma tran 2D, moi pixel la ma mau (0 -> n_colors-1)
        n_colors: Tong so mau sau luong tu hoa
    """
    h = img_hsv[:, :, 0].astype(np.int32)
    s = img_hsv[:, :, 1].astype(np.int32)
    v = img_hsv[:, :, 2].astype(np.int32)

    # OpenCV HSV: H in [0, 179], S in [0, 255], V in [0, 255]
    h_q = np.clip(h * h_bins // 180, 0, h_bins - 1).astype(np.int32)
    s_q = np.clip(s * s_bins // 256, 0, s_bins - 1).astype(np.int32)
    v_q = np.clip(v * v_bins // 256, 0, v_bins - 1).astype(np.int32)

    # Ma hoa thanh 1 so duy nhat cho moi to hop mau
    n_colors = h_bins * s_bins * v_bins
    quantized = h_q * (s_bins * v_bins) + s_q * v_bins + v_q

    return quantized, n_colors


def quantize_colors_rgb(img_bgr, bins=4):
    """Luong tu hoa mau sac trong khong gian RGB.

    Args:
        img_bgr: Anh BGR (numpy array)
        bins: So bin cho moi kenh R, G, B

    Returns:
        quantized: Ma tran 2D, moi pixel la ma mau
        n_colors: Tong so mau sau luong tu hoa
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[:, :, 0].astype(np.int32)
    g = img_rgb[:, :, 1].astype(np.int32)
    b = img_rgb[:, :, 2].astype(np.int32)

    r_q = np.clip(r * bins // 256, 0, bins - 1).astype(np.int32)
    g_q = np.clip(g * bins // 256, 0, bins - 1).astype(np.int32)
    b_q = np.clip(b * bins // 256, 0, bins - 1).astype(np.int32)

    n_colors = bins ** 3
    quantized = r_q * (bins * bins) + g_q * bins + b_q

    return quantized, n_colors


def load_dataset(data_dir, size=(128, 128)):
    """Tai toan bo dataset tu thu muc co cau truc class/image.

    Args:
        data_dir: Thu muc goc chua cac thu muc con (moi thu muc la 1 lop)
        size: Kich thuoc resize

    Returns:
        images: List cac anh BGR
        labels: List cac nhan (ten thu muc)
        paths: List duong dan file
    """
    data_dir = Path(data_dir)
    images = []
    labels = []
    paths = []

    # Sap xep de dam bao thu tu nhat quan
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if len(class_dirs) == 0:
        raise ValueError(f"Khong tim thay thu muc con nao trong {data_dir}")

    print(f"Tim thay {len(class_dirs)} lop:")
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Ho tro nhieu dinh dang anh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            image_files.extend(list(class_dir.glob(ext)))
            image_files.extend(list(class_dir.glob(ext.upper())))

        image_files = sorted(set(image_files))
        print(f"  - {class_name}: {len(image_files)} anh")

        for img_path in image_files:
            try:
                img = load_image(img_path, size)
                images.append(img)
                labels.append(class_name)
                paths.append(str(img_path))
            except Exception as e:
                print(f"    Loi khi doc {img_path}: {e}")

    print(f"\nTong cong: {len(images)} anh, {len(set(labels))} lop")
    return images, labels, paths


def get_class_names(data_dir):
    """Lay danh sach ten cac lop tu thu muc dataset.

    Args:
        data_dir: Thu muc goc chua dataset

    Returns:
        class_names: List ten cac lop (da sap xep)
    """
    data_dir = Path(data_dir)
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


if __name__ == "__main__":
    # Test nhanh
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corel-1k")

    print(f"Thu muc du lieu: {data_dir}")

    if os.path.exists(data_dir):
        images, labels, paths = load_dataset(data_dir)

        if len(images) > 0:
            # Test tien xu ly 1 anh
            img = images[0]
            print(f"\nTest anh dau tien: {paths[0]}")
            print(f"  Kich thuoc: {img.shape}")

            # Test HSV quantization
            img_hsv = convert_to_hsv(img)
            q_hsv, n_hsv = quantize_colors_hsv(img_hsv)
            print(f"  HSV quantized: {q_hsv.shape}, {n_hsv} mau")
            print(f"  So mau thuc te: {len(np.unique(q_hsv))}")

            # Test RGB quantization
            q_rgb, n_rgb = quantize_colors_rgb(img)
            print(f"  RGB quantized: {q_rgb.shape}, {n_rgb} mau")
            print(f"  So mau thuc te: {len(np.unique(q_rgb))}")
    else:
        print(f"Thu muc {data_dir} khong ton tai!")
        print("Hay tai dataset va dat vao thu muc data/corel-1k/")
