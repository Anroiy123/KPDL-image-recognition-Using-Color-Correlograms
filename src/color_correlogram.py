"""
color_correlogram.py - Cai dat thuat toan Color Auto-Correlogram

Tham khao: Huang et al. (1997) "Image Indexing Using Color Correlograms"

Auto-correlogram do xac suat 2 pixel CUNG MAU o khoang cach d:
    alpha(c, d) = Pr[p2 thuoc I_c | p1 thuoc I_c, |p1 - p2| = d]

Trong do:
    - c: ma mau (sau luong tu hoa)
    - d: khoang cach giua 2 pixel
    - I_c: tap cac pixel co mau c trong anh
"""

from typing import List, Optional, Tuple
import numpy as np


def _validate_quantized_image(quantized_img: np.ndarray, param_name: str = "quantized_img") -> None:
    """Validate quantized image array."""
    if not isinstance(quantized_img, np.ndarray):
        raise TypeError(f"{param_name} must be numpy array, got {type(quantized_img)}")
    if quantized_img.ndim != 2:
        raise ValueError(f"{param_name} must be 2D array, got shape {quantized_img.shape}")
    if quantized_img.dtype not in [np.int32, np.int64, np.int16, int]:
        raise TypeError(f"{param_name} must have integer dtype, got {quantized_img.dtype}")


def _validate_n_colors(n_colors: int, param_name: str = "n_colors") -> None:
    """Validate n_colors parameter."""
    if not isinstance(n_colors, (int, np.integer)) or n_colors <= 0:
        raise ValueError(f"{param_name} must be positive integer, got {n_colors}")


def _validate_distances(distances: Optional[List[int]], param_name: str = "distances") -> None:
    """Validate distances parameter."""
    if distances is not None:
        if not isinstance(distances, (list, tuple)):
            raise TypeError(f"{param_name} must be list/tuple or None, got {type(distances)}")
        if len(distances) == 0:
            raise ValueError(f"{param_name} must not be empty")
        for d in distances:
            if not isinstance(d, (int, np.integer)) or d <= 0:
                raise ValueError(f"{param_name} values must be positive integers, got {distances}")


def auto_correlogram(quantized_img: np.ndarray, n_colors: int, distances: Optional[List[int]] = None) -> np.ndarray:
    """Tinh Auto-Correlogram cho 1 anh da luong tu hoa.

    Su dung numpy array shifting de toi uu toc do (vectorized).
    KHONG dung vong for duyet tung pixel.

    Args:
        quantized_img: Ma tran 2D (H x W), moi pixel la ma mau (0 -> n_colors-1)
        n_colors: Tong so mau sau luong tu hoa
        distances: List cac khoang cach d can tinh. Mac dinh [1, 3, 5, 7]

    Returns:
        feature: Vector dac trung 1 chieu, kich thuoc (n_colors * len(distances),)

    Raises:
        TypeError: If quantized_img is not a valid array
        ValueError: If n_colors or distances are invalid
    """
    _validate_quantized_image(quantized_img)
    _validate_n_colors(n_colors)
    _validate_distances(distances)

    if distances is None:
        distances = [1, 3, 5, 7]

    h, w = quantized_img.shape
    correlogram = np.zeros((n_colors, len(distances)), dtype=np.float64)

    # Dem so pixel cua moi mau
    color_count = np.zeros(n_colors, dtype=np.float64)
    for c in range(n_colors):
        color_count[c] = np.sum(quantized_img == c)

    for d_idx, d in enumerate(distances):
        # 8 huong lan can o khoang cach d
        # (dy, dx) tuong ung voi 8 huong xung quanh
        shifts = [
            (-d, -d), (-d, 0), (-d, d),
            (0, -d),           (0, d),
            (d, -d),  (d, 0),  (d, d)
        ]

        # Dem so lan match cho moi mau
        match_count = np.zeros(n_colors, dtype=np.float64)
        neighbor_count = np.zeros(n_colors, dtype=np.float64)

        for dy, dx in shifts:
            # Xac dinh vung giao nhau giua anh goc va anh dich chuyen
            # Anh goc: quantized_img[y1_src:y2_src, x1_src:x2_src]
            # Anh dich chuyen: quantized_img[y1_dst:y2_dst, x1_dst:x2_dst]

            y1_src = max(0, -dy)
            y2_src = min(h, h - dy)
            x1_src = max(0, -dx)
            x2_src = min(w, w - dx)

            y1_dst = y1_src + dy
            y2_dst = y2_src + dy
            x1_dst = x1_src + dx
            x2_dst = x2_src + dx

            if y2_src <= y1_src or x2_src <= x1_src:
                continue

            center = quantized_img[y1_src:y2_src, x1_src:x2_src]
            neighbor = quantized_img[y1_dst:y2_dst, x1_dst:x2_dst]

            # So sanh: pixel trung tam va pixel lan can co cung mau khong?
            match = (center == neighbor)

            # Dem cho tung mau
            for c in range(n_colors):
                mask = (center == c)
                count = np.sum(mask)
                if count > 0:
                    neighbor_count[c] += count
                    match_count[c] += np.sum(match[mask])

        # Tinh xac suat: alpha(c, d) = match / neighbor
        for c in range(n_colors):
            if neighbor_count[c] > 0:
                correlogram[c, d_idx] = match_count[c] / neighbor_count[c]

    # Ghep thanh vector 1 chieu
    feature = correlogram.flatten()

    return feature


def auto_correlogram_fast(quantized_img: np.ndarray, n_colors: int, distances: Optional[List[int]] = None) -> np.ndarray:
    """Phien ban toi uu hon cua auto_correlogram.

    Thay vi lap qua tung mau, su dung one-hot encoding va phep nhan ma tran.
    Nhanh hon dang ke voi so luong mau lon.

    Args:
        quantized_img: Ma tran 2D (H x W)
        n_colors: Tong so mau
        distances: List khoang cach

    Returns:
        feature: Vector dac trung 1 chieu

    Raises:
        TypeError: If quantized_img is not a valid array
        ValueError: If n_colors or distances are invalid
    """
    _validate_quantized_image(quantized_img)
    _validate_n_colors(n_colors)
    _validate_distances(distances)

    if distances is None:
        distances = [1, 3, 5, 7]

    h, w = quantized_img.shape
    correlogram = np.zeros((n_colors, len(distances)), dtype=np.float64)

    for d_idx, d in enumerate(distances):
        shifts = [
            (-d, -d), (-d, 0), (-d, d),
            (0, -d),           (0, d),
            (d, -d),  (d, 0),  (d, d)
        ]

        total_match = np.zeros(n_colors, dtype=np.float64)
        total_count = np.zeros(n_colors, dtype=np.float64)

        for dy, dx in shifts:
            y1_src = max(0, -dy)
            y2_src = min(h, h - dy)
            x1_src = max(0, -dx)
            x2_src = min(w, w - dx)

            if y2_src <= y1_src or x2_src <= x1_src:
                continue

            center = quantized_img[y1_src:y2_src, x1_src:x2_src].ravel()
            neighbor = quantized_img[y1_src + dy:y2_src + dy,
                                     x1_src + dx:x2_src + dx].ravel()

            match = (center == neighbor)

            # Dem nhanh bang bincount
            total_count += np.bincount(center, minlength=n_colors).astype(np.float64)
            total_match += np.bincount(center[match], minlength=n_colors).astype(np.float64)

        # Tinh xac suat
        valid = total_count > 0
        correlogram[valid, d_idx] = total_match[valid] / total_count[valid]

    return correlogram.flatten()


def spatial_correlogram(quantized_img: np.ndarray, n_colors: int, distances: Optional[List[int]] = None, grid_size: int = 2, include_global: bool = True) -> np.ndarray:
    """Tinh correlogram co bo cuc khong gian don gian.

    Vector dau ra la phep noi:
    - Correlogram toan anh
    - Correlogram cua tung o trong luoi `grid_size x grid_size`

    Cach lam nay giu lai thong tin mau, dong thoi bo sung vi tri tuong doi
    de phan biet cac lop co mau gan nhau nhung bo cuc khac nhau.

    Args:
        quantized_img: Ma tran 2D (H x W)
        n_colors: Tong so mau
        distances: List khoang cach
        grid_size: Kich thuoc luoi spatial
        include_global: Co giu correlogram toan anh hay khong

    Returns:
        feature: Vector dac trung 1 chieu

    Raises:
        TypeError: If quantized_img is not a valid array
        ValueError: If n_colors, distances, or grid_size are invalid
    """
    _validate_quantized_image(quantized_img)
    _validate_n_colors(n_colors)
    _validate_distances(distances)
    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError(f"grid_size must be positive integer, got {grid_size}")

    features = []

    if include_global:
        features.append(auto_correlogram_fast(quantized_img, n_colors, distances))

    h, w = quantized_img.shape
    row_edges = np.linspace(0, h, grid_size + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, grid_size + 1, dtype=np.int32)

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            patch = quantized_img[
                row_edges[row_idx]:row_edges[row_idx + 1],
                col_edges[col_idx]:col_edges[col_idx + 1]
            ]
            features.append(auto_correlogram_fast(patch, n_colors, distances))

    return np.concatenate(features)


def _quantize_image_for_correlogram(img_bgr: np.ndarray, color_space: str = 'hsv',
                                    h_bins: int = 8, s_bins: int = 3, v_bins: int = 3,
                                    rgb_bins: int = 4) -> Tuple[np.ndarray, int]:
    """Luong tu hoa anh ve ma mau de tinh correlogram."""
    import cv2

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
        return quantized, n_colors

    if color_space == 'rgb':
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        n_colors = rgb_bins ** 3

        r = np.clip(img_rgb[:, :, 0].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        g = np.clip(img_rgb[:, :, 1].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        b = np.clip(img_rgb[:, :, 2].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        quantized = r * (rgb_bins * rgb_bins) + g * rgb_bins + b
        return quantized, n_colors

    raise ValueError(f"color_space phai la 'hsv' hoac 'rgb', khong phai '{color_space}'")


def extract_correlogram_feature(img_bgr: np.ndarray, color_space: str = 'hsv',
                                h_bins: int = 8, s_bins: int = 3, v_bins: int = 3,
                                rgb_bins: int = 4, distances: Optional[List[int]] = None,
                                spatial_grid: Optional[int] = None, include_global: bool = True) -> np.ndarray:
    """Ham tien ich: tu anh BGR goc -> vector dac trung correlogram.

    Args:
        img_bgr: Anh BGR (numpy array)
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin cho HSV
        rgb_bins: So bin cho RGB
        distances: List khoang cach
        spatial_grid: Neu khac None, noi them correlogram cua luoi spatial_grid x spatial_grid
        include_global: Co giu correlogram toan anh o dau vector hay khong

    Returns:
        feature: Vector dac trung 1 chieu

    Raises:
        ValueError: If color_space is invalid
    """
    quantized, n_colors = _quantize_image_for_correlogram(
        img_bgr,
        color_space=color_space,
        h_bins=h_bins,
        s_bins=s_bins,
        v_bins=v_bins,
        rgb_bins=rgb_bins,
    )

    if spatial_grid is not None:
        return spatial_correlogram(
            quantized,
            n_colors,
            distances=distances,
            grid_size=spatial_grid,
            include_global=include_global,
        )

    return auto_correlogram_fast(quantized, n_colors, distances)
    """Tinh Auto-Correlogram cho 1 anh da luong tu hoa.

    Su dung numpy array shifting de toi uu toc do (vectorized).
    KHONG dung vong for duyet tung pixel.

    Args:
        quantized_img: Ma tran 2D (H x W), moi pixel la ma mau (0 -> n_colors-1)
        n_colors: Tong so mau sau luong tu hoa
        distances: List cac khoang cach d can tinh. Mac dinh [1, 3, 5, 7]

    Returns:
        feature: Vector dac trung 1 chieu, kich thuoc (n_colors * len(distances),)
    """
    if distances is None:
        distances = [1, 3, 5, 7]

    h, w = quantized_img.shape
    correlogram = np.zeros((n_colors, len(distances)), dtype=np.float64)

    # Dem so pixel cua moi mau
    color_count = np.zeros(n_colors, dtype=np.float64)
    for c in range(n_colors):
        color_count[c] = np.sum(quantized_img == c)

    for d_idx, d in enumerate(distances):
        # 8 huong lan can o khoang cach d
        # (dy, dx) tuong ung voi 8 huong xung quanh
        shifts = [
            (-d, -d), (-d, 0), (-d, d),
            (0, -d),           (0, d),
            (d, -d),  (d, 0),  (d, d)
        ]

        # Dem so lan match cho moi mau
        match_count = np.zeros(n_colors, dtype=np.float64)
        neighbor_count = np.zeros(n_colors, dtype=np.float64)

        for dy, dx in shifts:
            # Xac dinh vung giao nhau giua anh goc va anh dich chuyen
            # Anh goc: quantized_img[y1_src:y2_src, x1_src:x2_src]
            # Anh dich chuyen: quantized_img[y1_dst:y2_dst, x1_dst:x2_dst]

            y1_src = max(0, -dy)
            y2_src = min(h, h - dy)
            x1_src = max(0, -dx)
            x2_src = min(w, w - dx)

            y1_dst = y1_src + dy
            y2_dst = y2_src + dy
            x1_dst = x1_src + dx
            x2_dst = x2_src + dx

            if y2_src <= y1_src or x2_src <= x1_src:
                continue

            center = quantized_img[y1_src:y2_src, x1_src:x2_src]
            neighbor = quantized_img[y1_dst:y2_dst, x1_dst:x2_dst]

            # So sanh: pixel trung tam va pixel lan can co cung mau khong?
            match = (center == neighbor)

            # Dem cho tung mau
            for c in range(n_colors):
                mask = (center == c)
                count = np.sum(mask)
                if count > 0:
                    neighbor_count[c] += count
                    match_count[c] += np.sum(match[mask])

        # Tinh xac suat: alpha(c, d) = match / neighbor
        for c in range(n_colors):
            if neighbor_count[c] > 0:
                correlogram[c, d_idx] = match_count[c] / neighbor_count[c]

    # Ghep thanh vector 1 chieu
    feature = correlogram.flatten()

    return feature


def auto_correlogram_fast(quantized_img, n_colors, distances=None):
    """Phien ban toi uu hon cua auto_correlogram.

    Thay vi lap qua tung mau, su dung one-hot encoding va phep nhan ma tran.
    Nhanh hon dang ke voi so luong mau lon.

    Args:
        quantized_img: Ma tran 2D (H x W)
        n_colors: Tong so mau
        distances: List khoang cach

    Returns:
        feature: Vector dac trung 1 chieu
    """
    if distances is None:
        distances = [1, 3, 5, 7]

    h, w = quantized_img.shape
    correlogram = np.zeros((n_colors, len(distances)), dtype=np.float64)

    for d_idx, d in enumerate(distances):
        shifts = [
            (-d, -d), (-d, 0), (-d, d),
            (0, -d),           (0, d),
            (d, -d),  (d, 0),  (d, d)
        ]

        total_match = np.zeros(n_colors, dtype=np.float64)
        total_count = np.zeros(n_colors, dtype=np.float64)

        for dy, dx in shifts:
            y1_src = max(0, -dy)
            y2_src = min(h, h - dy)
            x1_src = max(0, -dx)
            x2_src = min(w, w - dx)

            if y2_src <= y1_src or x2_src <= x1_src:
                continue

            center = quantized_img[y1_src:y2_src, x1_src:x2_src].ravel()
            neighbor = quantized_img[y1_src + dy:y2_src + dy,
                                     x1_src + dx:x2_src + dx].ravel()

            match = (center == neighbor)

            # Dem nhanh bang bincount
            total_count += np.bincount(center, minlength=n_colors).astype(np.float64)
            total_match += np.bincount(center[match], minlength=n_colors).astype(np.float64)

        # Tinh xac suat
        valid = total_count > 0
        correlogram[valid, d_idx] = total_match[valid] / total_count[valid]

    return correlogram.flatten()


def spatial_correlogram(quantized_img, n_colors, distances=None, grid_size=2, include_global=True):
    """Tinh correlogram co bo cuc khong gian don gian.

    Vector dau ra la phep noi:
    - Correlogram toan anh
    - Correlogram cua tung o trong luoi `grid_size x grid_size`

    Cach lam nay giu lai thong tin mau, dong thoi bo sung vi tri tuong doi
    de phan biet cac lop co mau gan nhau nhung bo cuc khac nhau.
    """
    features = []

    if include_global:
        features.append(auto_correlogram_fast(quantized_img, n_colors, distances))

    h, w = quantized_img.shape
    row_edges = np.linspace(0, h, grid_size + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, grid_size + 1, dtype=np.int32)

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            patch = quantized_img[
                row_edges[row_idx]:row_edges[row_idx + 1],
                col_edges[col_idx]:col_edges[col_idx + 1]
            ]
            features.append(auto_correlogram_fast(patch, n_colors, distances))

    return np.concatenate(features)


def _quantize_image_for_correlogram(img_bgr, color_space='hsv',
                                    h_bins=8, s_bins=3, v_bins=3,
                                    rgb_bins=4):
    """Luong tu hoa anh ve ma mau de tinh correlogram."""
    import cv2

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
        return quantized, n_colors

    if color_space == 'rgb':
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        n_colors = rgb_bins ** 3

        r = np.clip(img_rgb[:, :, 0].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        g = np.clip(img_rgb[:, :, 1].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        b = np.clip(img_rgb[:, :, 2].astype(np.int32) * rgb_bins // 256, 0, rgb_bins - 1).astype(np.int32)
        quantized = r * (rgb_bins * rgb_bins) + g * rgb_bins + b
        return quantized, n_colors

    raise ValueError(f"color_space phai la 'hsv' hoac 'rgb', khong phai '{color_space}'")


def extract_correlogram_feature(img_bgr, color_space='hsv',
                                h_bins=8, s_bins=3, v_bins=3,
                                rgb_bins=4, distances=None,
                                spatial_grid=None, include_global=True):
    """Ham tien ich: tu anh BGR goc -> vector dac trung correlogram.

    Args:
        img_bgr: Anh BGR (numpy array)
        color_space: 'hsv' hoac 'rgb'
        h_bins, s_bins, v_bins: So bin cho HSV
        rgb_bins: So bin cho RGB
        distances: List khoang cach
        spatial_grid: Neu khac None, noi them correlogram cua luoi spatial_grid x spatial_grid
        include_global: Co giu correlogram toan anh o dau vector hay khong

    Returns:
        feature: Vector dac trung 1 chieu
    """
    quantized, n_colors = _quantize_image_for_correlogram(
        img_bgr,
        color_space=color_space,
        h_bins=h_bins,
        s_bins=s_bins,
        v_bins=v_bins,
        rgb_bins=rgb_bins,
    )

    if spatial_grid is not None:
        return spatial_correlogram(
            quantized,
            n_colors,
            distances=distances,
            grid_size=spatial_grid,
            include_global=include_global,
        )

    return auto_correlogram_fast(quantized, n_colors, distances)


if __name__ == "__main__":
    # Test nhanh voi 1 anh
    import cv2
    import sys
    import time

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Su dung: python color_correlogram.py <duong_dan_anh>")
        print("Dang tao anh test ngau nhien...")
        # Tao anh test
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img_path = None

    if img_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))

    print(f"Kich thuoc anh: {img.shape}")

    # Test phien ban thuong
    start = time.time()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0].astype(np.int32)
    s = img_hsv[:, :, 1].astype(np.int32)
    v = img_hsv[:, :, 2].astype(np.int32)
    h_q = np.clip(h * 8 // 180, 0, 7).astype(np.int32)
    s_q = np.clip(s * 3 // 256, 0, 2).astype(np.int32)
    v_q = np.clip(v * 3 // 256, 0, 2).astype(np.int32)
    quantized = h_q * 9 + s_q * 3 + v_q

    feat1 = auto_correlogram(quantized, 72)
    t1 = time.time() - start
    print(f"auto_correlogram:      {t1:.3f}s, vector size: {feat1.shape}")

    # Test phien ban nhanh
    start = time.time()
    feat2 = auto_correlogram_fast(quantized, 72)
    t2 = time.time() - start
    print(f"auto_correlogram_fast: {t2:.3f}s, vector size: {feat2.shape}")

    # Test ham tien ich
    start = time.time()
    feat3 = extract_correlogram_feature(img, color_space='hsv')
    t3 = time.time() - start
    print(f"extract_feature (HSV): {t3:.3f}s, vector size: {feat3.shape}")

    feat4 = extract_correlogram_feature(img, color_space='rgb')
    print(f"extract_feature (RGB): vector size: {feat4.shape}")

    print(f"\nToc do cai thien: {t1/t2:.1f}x")
    print(f"Vector HSV: min={feat3.min():.4f}, max={feat3.max():.4f}, mean={feat3.mean():.4f}")
