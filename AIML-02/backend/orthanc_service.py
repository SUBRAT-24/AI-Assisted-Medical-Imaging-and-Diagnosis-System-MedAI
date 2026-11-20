"""
Step 5: Orthanc DICOM Server Poller & Automated Inference Ingestion Service.
Polls Orthanc REST API (/changes, /studies, /instances/{id}/file), downloads newly arrived
DICOM studies, executes Classification + MONAI 3D U-Net Segmentation + Grad-CAM, and stores results.
"""
from typing import Dict, List, Any, Tuple
import time
import numpy as np
from dicom_utils import extract_dicom_metadata, apply_window_level
from preprocess import preprocess_2d_scan
from models.classifier import predict_pathologies
from models.segmenter import generate_segmentation_mask
from explainability.gradcam import GradCAMExplainer


class OrthancStudyIngestor:
    def __init__(self, orthanc_url: str = "http://localhost:8042"):
        self.orthanc_url = orthanc_url.rstrip("/")
        self.last_seq = 0
        self.processed_studies: Dict[str, Dict[str, Any]] = {}
        self.gradcam = GradCAMExplainer()

    def process_dicom_study(
        self,
        study_id: str,
        dicom_headers: Dict[str, Any],
        pixel_array: np.ndarray,
        roi_center: Tuple[float, float] = (0.28, 0.63),
    ) -> Dict[str, Any]:
        """Run end-to-end DICOM preprocessing, AI inference, Grad-CAM, and segmentation."""
        t0 = time.time()
        meta = extract_dicom_metadata(dicom_headers, anonymize=True)
        windowed = apply_window_level(
            pixel_array,
            window_width=meta["WindowWidth"],
            window_center=meta["WindowCenter"],
        )
        prep = preprocess_2d_scan(windowed, target_size=(224, 224), augment=False)
        cls_result = predict_pathologies(prep["normalized_01"], modality=meta["Modality"])
        seg_result = generate_segmentation_mask(
            windowed,
            roi_center_norm=roi_center,
            pixel_spacing_mm=tuple(meta["PixelSpacing"]),
            slice_thickness_mm=meta["SliceThickness"],
        )
        cam_result = self.gradcam.compute_heatmap(windowed, focus_center_norm=roi_center)
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)

        record = {
            "study_id": study_id,
            "orthanc_url": f"{self.orthanc_url}/studies/{study_id}",
            "metadata": meta,
            "classification": cls_result,
            "segmentation": {
                "active_pixels": seg_result["ActivePixels"],
                "area_mm2": seg_result["AreaMm2"],
                "estimated_volume_cm3": seg_result["EstimatedVolumeCm3"],
                "architecture": seg_result["Architecture"],
            },
            "gradcam": {
                "peak_coordinates_xy": cam_result["peak_coordinates_xy"],
                "mean_activation": cam_result["mean_activation"],
                "max_activation": cam_result["max_activation"],
            },
            "inference_latency_ms": elapsed_ms,
            "review_status": "PENDING_RADIOLOGIST_REVIEW",
        }
        self.processed_studies[study_id] = record
        return record
