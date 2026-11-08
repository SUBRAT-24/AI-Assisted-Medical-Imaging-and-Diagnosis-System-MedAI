"""
Automated Unit & Integration Test Suite for AIML-02 Medical Imaging & Diagnosis System.
Tests all 7 project stages:
1. DICOM Metadata Extraction, PHI Anonymization & HU Windowing
2. Image Preprocessing (Resize, Normalization, Augmentation)
3. Classification (DenseNet/EfficientNet) & 3D U-Net Segmentation Metrics (Dice/IoU)
4. Grad-CAM Spatial Heatmap Attribution
5. Orthanc DICOM Study Ingestion Service
6. Radiologist Approve/Reject Immutable Audit Log Persistence
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from dicom_utils import apply_hu_rescaling, apply_window_level, extract_dicom_metadata
from preprocess import preprocess_2d_scan
from models.classifier import predict_pathologies
from models.segmenter import compute_dice_iou, generate_segmentation_mask
from explainability.gradcam import GradCAMExplainer
from orthanc_service import OrthancStudyIngestor
from main import AuditLogger, RadiologistDecisionRequest


class TestMedicalImagingPipeline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(101)
        self.synthetic_scan = rng.integers(0, 2000, size=(256, 256), dtype=np.int16)
        # Inject hyperdense focal region at right lower zone
        self.synthetic_scan[140:185, 55:100] = 2800

    def test_01_dicom_hu_and_anonymization(self):
        hu = apply_hu_rescaling(self.synthetic_scan, slope=1.0, intercept=-1024.0)
        self.assertEqual(hu.shape, (256, 256))
        self.assertAlmostEqual(float(hu.min()), -1024.0, places=1)

        windowed = apply_window_level(hu, window_width=1500.0, window_center=-600.0)
        self.assertEqual(windowed.dtype, np.uint8)
        self.assertTrue(0 <= windowed.min() <= windowed.max() <= 255)

        raw_tags = {
            "PatientName": "John Doe",
            "PatientID": "PT-99281",
            "Modality": "CT",
            "WindowCenter": -600,
            "WindowWidth": 1500,
        }
        meta = extract_dicom_metadata(raw_tags, anonymize=True)
        self.assertEqual(meta["PatientName"], "ANONYMIZED")
        self.assertEqual(meta["Modality"], "CT")
        self.assertIn("FOR EDUCATION ONLY", meta["Disclaimer"])

    def test_02_preprocessing_and_augmentation(self):
        res_clean = preprocess_2d_scan(self.synthetic_scan, target_size=(224, 224), augment=False)
        self.assertEqual(res_clean["shape"], [224, 224])
        self.assertAlmostEqual(float(res_clean["normalized_01"].min()), 0.0, places=4)
        self.assertAlmostEqual(float(res_clean["normalized_01"].max()), 1.0, places=4)

        res_aug = preprocess_2d_scan(self.synthetic_scan, target_size=(224, 224), augment=True, seed=7)
        self.assertEqual(res_aug["shape"], [224, 224])

    def test_03_classification_and_segmentation(self):
        preds = predict_pathologies(self.synthetic_scan, modality="CR")
        self.assertIn("primary_finding", preds)
        self.assertGreater(preds["primary_confidence"], 0.50)
        self.assertGreaterEqual(len(preds["predictions"]), 6)

        seg = generate_segmentation_mask(self.synthetic_scan, roi_center_norm=(0.30, 0.63))
        self.assertEqual(seg["mask"].shape, (256, 256))
        self.assertGreater(seg["ActivePixels"], 50)
        self.assertGreater(seg["EstimatedVolumeCm3"], 0.0)

        metrics = compute_dice_iou(seg["mask"], seg["mask"])
        self.assertAlmostEqual(metrics["dice"], 1.0, places=3)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=3)

    def test_04_gradcam_explainability(self):
        explainer = GradCAMExplainer()
        cam = explainer.compute_heatmap(self.synthetic_scan, focus_center_norm=(0.30, 0.63))
        self.assertEqual(cam["heatmap_norm"].shape, (256, 256))
        self.assertEqual(cam["heatmap_rgb"].shape, (256, 256, 3))
        self.assertAlmostEqual(cam["max_activation"], 1.0, places=3)

    def test_05_orthanc_study_ingestor(self):
        ingestor = OrthancStudyIngestor(orthanc_url="http://orthanc-pacs:8042")
        study_rec = ingestor.process_dicom_study(
            study_id="ORTHANC-STUDY-001",
            dicom_headers={"PatientID": "PT-104", "Modality": "CR", "WindowCenter": 500, "WindowWidth": 2000},
            pixel_array=self.synthetic_scan,
        )
        self.assertEqual(study_rec["study_id"], "ORTHANC-STUDY-001")
        self.assertEqual(study_rec["review_status"], "PENDING_RADIOLOGIST_REVIEW")
        self.assertIn("classification", study_rec)
        self.assertIn("segmentation", study_rec)
        self.assertIn("gradcam", study_rec)

    def test_06_radiologist_audit_logger(self):
        logger = AuditLogger(db_path=":memory:")
        req_approve = RadiologistDecisionRequest(
            study_id="ORTHANC-STUDY-001",
            patient_id="PT-104",
            modality="CR",
            ai_finding="Pneumonia",
            ai_confidence=0.912,
            decision="APPROVED",
            radiologist_name="Dr. A. Sharma, MD (Radiology)",
            radiologist_notes="Confirmed right lower lobe air-space consolidation.",
        )
        saved = logger.record_decision(req_approve)
        self.assertEqual(saved["decision"], "APPROVED")
        logs = logger.list_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["radiologist_name"], "Dr. A. Sharma, MD (Radiology)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
