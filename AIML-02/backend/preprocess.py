"""
Step 2: Medical Image Preprocessing & Augmentation Pipeline (2D CXR & 3D CT/MRI).
Implements resize, z-score / min-max intensity normalization, and stochastic data augmentation.
Integrates MONAI dictionary transforms when installed and NumPy/SciPy fallback for CI testing.
"""
from typing import Tuple, Dict, Any
import numpy as np
import cv2

try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Spacingd,
        Orientationd,
        ScaleIntensityRanged,
        CropForegroundd,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        ResizeWithPadOrCropd,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


def get_monai_3d_transforms(patch_size: Tuple[int, int, int] = (128, 128, 64), is_train: bool = True):
    """Build MONAI 3D volumetric transform pipeline for CT/MRI segmentation."""
    if not HAS_MONAI:
        return None
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=400.0, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),
    ]
    if is_train:
        base_transforms.extend([
            RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        ])
    return Compose(base_transforms)


def preprocess_2d_scan(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Resize, normalize, and optionally augment a 2D medical radiograph/slice.
    Returns normalized float32 tensor array in [0, 1] and z-score standardized array.
    """
    rng = np.random.default_rng(seed)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image.copy()

    resized = cv2.resize(gray.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
    min_val, max_val = float(np.min(resized)), float(np.max(resized))
    normalized = (resized - min_val) / (max_val - min_val + 1e-8)

    if augment:
        # Random horizontal flip
        if rng.random() > 0.5:
            normalized = np.fliplr(normalized).copy()
        # Small random rotation (-10 to +10 deg)
        angle = float(rng.uniform(-10.0, 10.0))
        center = (target_size[0] // 2, target_size[1] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        normalized = cv2.warpAffine(normalized, rot_mat, target_size, borderMode=cv2.BORDER_REFLECT)
        # Mild Gaussian noise
        noise = rng.normal(0.0, 0.02, size=normalized.shape).astype(np.float32)
        normalized = np.clip(normalized + noise, 0.0, 1.0)

    mean = float(np.mean(normalized))
    std = float(np.std(normalized)) + 1e-8
    z_score = (normalized - mean) / std

    return {
        "normalized_01": normalized.astype(np.float32),
        "standardized_z": z_score.astype(np.float32),
        "shape": list(normalized.shape),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }
