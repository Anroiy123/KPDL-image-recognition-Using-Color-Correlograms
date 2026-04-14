"""
dataset_profile.py - Cau hinh profile dataset va helper artifact namespace.
"""

from pathlib import Path


DEFAULT_DATASET_PROFILE = "corel-1k"


DATASET_PROFILES = {
    "corel-1k": {
        "dataset_name": "corel-1k",
        "display_name": "Corel-1K",
        "data_dir": Path("data") / "corel-1k",
        "split_filename": "corel-1k_split.json",
    },
    "ucmerced-landuse": {
        "dataset_name": "ucmerced-landuse",
        "display_name": "UCMerced LandUse",
        "data_dir": Path("data") / "UCMerced_LandUse" / "Images",
        "split_filename": "ucmerced-landuse_split.json",
    },
}


def list_dataset_profiles():
    """Tra ve danh sach profile key theo thu tu khai bao."""
    return list(DATASET_PROFILES.keys())


def validate_dataset_profile(profile_key):
    """Xac thuc profile key va tra ve key hop le."""
    if profile_key in DATASET_PROFILES:
        return profile_key

    available = ", ".join(list_dataset_profiles())
    raise ValueError(
        f"Dataset profile khong hop le: '{profile_key}'. "
        f"Ho tro: {available}"
    )


def get_profile_config(profile_key):
    """Lay cau hinh profile theo key."""
    key = validate_dataset_profile(profile_key)
    return DATASET_PROFILES[key]


def get_split_filename(profile_key):
    """Lay ten file split mac dinh theo profile."""
    return get_profile_config(profile_key)["split_filename"]


def resolve_dataset_profile(profile_key, project_dir):
    """Resolve profile thanh cac duong dan tuyet doi trong project."""
    key = validate_dataset_profile(profile_key)
    cfg = DATASET_PROFILES[key]
    project_dir = Path(project_dir)

    resolved = {
        "key": key,
        "dataset_name": cfg["dataset_name"],
        "display_name": cfg.get("display_name", cfg["dataset_name"]),
        "data_dir": project_dir / cfg["data_dir"],
        "split_filename": cfg["split_filename"],
    }
    return resolved


def scoped_artifact_name(profile_key, base_name):
    """Them namespace profile vao ten artifact."""
    key = validate_dataset_profile(profile_key)
    return f"{key}_{base_name}"


def scoped_artifact_path(directory, profile_key, base_name):
    """Tra ve duong dan artifact da namespace theo profile."""
    return Path(directory) / scoped_artifact_name(profile_key, base_name)


def resolve_scoped_or_legacy_path(directory, profile_key, base_name, allow_legacy_corel=True):
    """
    Tim artifact theo namespace profile. Neu khong co va la corel thi cho phep fallback ten cu.

    Returns:
        (path, is_legacy)
    """
    directory = Path(directory)
    scoped_path = scoped_artifact_path(directory, profile_key, base_name)
    if scoped_path.exists():
        return scoped_path, False

    legacy_path = directory / base_name
    if allow_legacy_corel and profile_key == DEFAULT_DATASET_PROFILE and legacy_path.exists():
        return legacy_path, True

    return scoped_path, False
