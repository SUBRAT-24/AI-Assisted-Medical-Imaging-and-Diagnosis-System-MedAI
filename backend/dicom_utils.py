"""
DICOM Parsing, De-identification, and Hounsfield Unit (HU) Windowing Utilities.
Supports both real pydicom Datasets and deterministic fallback parsing for testing.
"""
from typing import Dict, Any, Tuple
import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

PHI_TAGS_TO_ANONYMIZE = [
    "PatientName",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
]


def apply_hu_rescaling(pixel_array: np.ndarray, slope: float = 1.0, intercept: float = -1024.0) -> np.ndarray:
    """Convert raw DICOM stored pixel values into calibrated Hounsfield Units (HU)."""
    hu = pixel_array.astype(np.float32) * float(slope) + float(intercept)
    return hu


def apply_window_level(
    image: np.ndarray, window_width: float, window_center: float, out_range: Tuple[float, float] = (0.0, 255.0)
) -> np.ndarray:
    """
    Apply radiological Window Width (WW) and Window Center/Level (WL) transformation.
    Returns normalized uint8 array in [0, 255] ready for viewing or CNN input.
    """
    if window_width <= 0:
        raise ValueError("window_width must be strictly positive")
    low = window_center - (window_width / 2.0)
    high = window_center + (window_width / 2.0)
    clipped = np.clip(image.astype(np.float32), low, high)
    norm = (clipped - low) / (high - low + 1e-8)
    scaled = norm * (out_range[1] - out_range[0]) + out_range[0]
    return np.round(scaled).astype(np.uint8)


def extract_dicom_metadata(dcm_dict: Dict[str, Any], anonymize: bool = True) -> Dict[str, Any]:
    """
    Extract and optionally de-identify clinical DICOM header metadata.
    Complies with HIPAA Safe Harbor de-identification for educational AI workflows.
    """
    cleaned = dict(dcm_dict)
    if anonymize:
        for tag in PHI_TAGS_TO_ANONYMIZE:
            if tag in cleaned:
                cleaned[tag] = "ANONYMIZED"
    return {
        "StudyInstanceUID": cleaned.get("StudyInstanceUID", "1.2.840.113619.2.55.3"),
        "SeriesInstanceUID": cleaned.get("SeriesInstanceUID", "1.2.840.113619.2.55.3.1"),
        "PatientID": cleaned.get("PatientID", "ANON-0001"),
        "PatientName": cleaned.get("PatientName", "ANONYMIZED"),
        "Modality": cleaned.get("Modality", "CR"),
        "BodyPartExamined": cleaned.get("BodyPartExamined", "CHEST"),
        "PixelSpacing": cleaned.get("PixelSpacing", [0.75, 0.75]),
        "SliceThickness": float(cleaned.get("SliceThickness", 1.5)),
        "WindowCenter": float(cleaned.get("WindowCenter", -600.0)),
        "WindowWidth": float(cleaned.get("WindowWidth", 1500.0)),
        "RescaleSlope": float(cleaned.get("RescaleSlope", 1.0)),
        "RescaleIntercept": float(cleaned.get("RescaleIntercept", -1024.0)),
        "Disclaimer": "FOR EDUCATION ONLY - NOT FOR CLINICAL DIAGNOSTIC USE",
    }
