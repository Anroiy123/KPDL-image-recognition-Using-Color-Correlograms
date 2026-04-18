"""
preprocessing.py - Tien xu ly anh cho bai toan Color Correlogram
"""

import os
from typing import List, Tuple, Union
import cv2
import numpy as np
from pathlib import Path


def _validate_image_array(img: np.ndarray, param_name: str = "img") -> None:
    """Validate image is a valid numpy array."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{param_name} must be numpy array, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"{param_name} must be 3D array (H, W, C), got shape {img.shape}")
    if img.shape[2] != 3:
        raise ValueError(f"{param_name} must have 3 channels, got {img.shape[2]}")


def _validate_size_tuple(size: Union[Tuple[int, int], List[int]], param_name: str = "size") -> None:
    """Validate size is a valid (width, height) tuple."""
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"{param_name} must be tuple/list of 2 ints, got {size}")
    if not all(isinstance(s, int) and s > 0 for s in size):
        raise ValueError(f"{param_name} values must be positive integers, got {size}")


def _validate_bins(h_bins: int, s_bins: int, v_bins: int) -> None:
    """Validate bin parameters."""
    for bins, name in [(h_bins, "h_bins"), (s_bins, "s_bins"), (v_bins, "v_bins")]:
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError(f"{name} must be positive integer, got {bins}")


def load_image(path: Union[str, Path], size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Doc anh va resize ve kich thuoc dong nhat.

    Args:
        path: Duong dan den file anh
        size: Kich thuoc dau ra (width, height)

    Returns:
        img: Anh BGR (numpy array) da resize

    Raises:
        ValueError: If image cannot be loaded or size is invalid
        TypeError: If path is not a valid path type
    """
    try:
        _validate_size_tuple(size, "size")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Khong the doc anh: {path}")
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img
    except (OSError, IOError) as e:
        raise ValueError(f"Loi doc file anh {path}: {e}")


def convert_to_hsv(img: np.ndarray) -> np.ndarray:
    """Chuyen anh tu BGR sang HSV.

    Args:
        img: Anh BGR (numpy array)

    Returns:
        img_hsv: Anh HSV (numpy array)

    Raises:
        TypeError: If img is not a valid image array
        ValueError: If img has invalid shape
    """
    _validate_image_array(img, "img")
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        raise ValueError(f"Loi chuyen doi BGR->HSV: {e}")


def quantize_colors_hsv(img_hsv: np.ndarray, h_bins: int = 8, s_bins: int = 3, v_bins: int = 3) -> Tuple[np.ndarray, int]:
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

    Raises:
        TypeError: If img_hsv is not a valid image array
        ValueError: If bins are invalid
    """
    _validate_image_array(img_hsv, "img_hsv")
    _validate_bins(h_bins, s_bins, v_bins)

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


def quantize_colors_rgb(img_bgr: np.ndarray, bins: int = 4) -> Tuple[np.ndarray, int]:
    """Luong tu hoa mau sac trong khong gian RGB.

    Args:
        img_bgr: Anh BGR (numpy array)
        bins: So bin cho moi kenh R, G, B

    Returns:
        quantized: Ma tran 2D, moi pixel la ma mau
        n_colors: Tong so mau sau luong tu hoa

    Raises:
        TypeError: If img_bgr is not a valid image array
        ValueError: If bins is invalid
    """
    _validate_image_array(img_bgr, "img_bgr")
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f"bins must be positive integer, got {bins}")

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        raise ValueError(f"Loi chuyen doi BGR->RGB: {e}")

    r = img_rgb[:, :, 0].astype(np.int32)
    g = img_rgb[:, :, 1].astype(np.int32)
    b = img_rgb[:, :, 2].astype(np.int32)

    r_q = np.clip(r * bins // 256, 0, bins - 1).astype(np.int32)
    g_q = np.clip(g * bins // 256, 0, bins - 1).astype(np.int32)
    b_q = np.clip(b * bins // 256, 0, bins - 1).astype(np.int32)

    n_colors = bins ** 3
    quantized = r_q * (bins * bins) + g_q * bins + b_q

    return quantized, n_colors


def load_dataset(data_dir: Union[str, Path], size: Tuple[int, int] = (128, 128)) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Tai toan bo dataset tu thu muc co cau truc class/image.

    Args:
        data_dir: Thu muc goc chua cac thu muc con (moi thu muc la 1 lop)
        size: Kich thuoc resize

    Returns:
        images: List cac anh BGR
        labels: List cac nhan (ten thu muc)
        paths: List duong dan file

    Raises:
        ValueError: If data_dir is invalid or empty
        TypeError: If size is invalid
    """
    _validate_size_tuple(size, "size")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Thu muc {data_dir} khong ton tai")
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} khong phai la thu muc")

    images: List[np.ndarray] = []
    labels: List[str] = []
    paths: List[str] = []

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


def get_class_names(data_dir: Union[str, Path]) -> List[str]:
    """Lay danh sach ten cac lop tu thu muc dataset.

    Args:
        data_dir: Thu muc goc chua dataset

    Returns:
        class_names: List ten cac lop (da sap xep)

    Raises:
        ValueError: If data_dir is invalid
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Thu muc {data_dir} khong ton tai")
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} khong phai la thu muc")

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
