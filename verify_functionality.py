#!/usr/bin/env python3
"""
Comprehensive functionality verification script for AIML-02 Medical Imaging System.
Tests all components without requiring external dependencies like Docker or test data.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    modules = [
        'dicom_utils',
        'preprocess',
        'models.classifier',
        'models.segmenter',
        'explainability.gradcam',
        'orthanc_service',
        'main',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    print("\n✓ All modules imported successfully\n")
    return True

def test_dicom_processing():
    """Test DICOM utilities."""
    print("=" * 70)
    print("TEST 2: DICOM Processing")
    print("=" * 70)
    
    try:
        import numpy as np
        from dicom_utils import apply_hu_rescaling, apply_window_level, extract_dicom_metadata
        
        # Create synthetic scan
        scan = np.random.randint(0, 2000, size=(256, 256), dtype=np.int16)
        
        # Test HU rescaling
        hu = apply_hu_rescaling(scan, slope=1.0, intercept=-1024.0)
        assert hu.shape == (256, 256), "HU rescaling shape mismatch"
        assert hu.min() >= -1024.0, "HU min value incorrect"
        print(f"✓ HU rescaling: min={hu.min():.1f}, max={hu.max():.1f}")
        
        # Test windowing
        windowed = apply_window_level(hu, window_width=1500.0, window_center=-600.0)
        assert windowed.dtype == np.uint8, "Windowed dtype incorrect"
        assert windowed.min() >= 0 and windowed.max() <= 255, "Windowed range incorrect"
        print(f"✓ Window/Level transform: shape={windowed.shape}, dtype={windowed.dtype}")
        
        # Test metadata extraction
        raw_tags = {
            "PatientName": "John Doe",
            "PatientID": "PT-12345",
            "Modality": "CT",
            "WindowCenter": -600,
            "WindowWidth": 1500,
        }
        meta = extract_dicom_metadata(raw_tags, anonymize=True)
        assert meta["PatientName"] == "ANONYMIZED", "Anonymization failed"
        assert "Disclaimer" in meta, "Disclaimer missing"
        print(f"✓ Metadata extraction: Modality={meta['Modality']}, Anonymized={meta['PatientName']}")
        
        print("\n✓ DICOM processing tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ DICOM processing failed: {e}\n")
        return False

def test_preprocessing():
    """Test image preprocessing."""
    print("=" * 70)
    print("TEST 3: Image Preprocessing")
    print("=" * 70)
    
    try:
        import numpy as np
        from preprocess import preprocess_2d_scan
        
        scan = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
        
        # Test without augmentation
        result = preprocess_2d_scan(scan, target_size=(224, 224), augment=False)
        assert result["shape"] == [224, 224], "Resize failed"
        assert result["normalized_01"].min() >= 0.0, "Normalization min failed"
        assert result["normalized_01"].max() <= 1.0, "Normalization max failed"
        print(f"✓ Resize: {result['shape']}")
        print(f"✓ Normalization: min={result['normalized_01'].min():.4f}, max={result['normalized_01'].max():.4f}")
        
        # Test with augmentation
        result_aug = preprocess_2d_scan(scan, target_size=(224, 224), augment=True, seed=42)
        assert result_aug["shape"] == [224, 224], "Augmented resize failed"
        print(f"✓ Augmentation: shape={result_aug['shape']}, mean={result_aug['mean']:.4f}")
        
        print("\n✓ Preprocessing tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}\n")
        return False

def test_classification():
    """Test pathology classification."""
    print("=" * 70)
    print("TEST 4: Pathology Classification")
    print("=" * 70)
    
    try:
        import numpy as np
        from models.classifier import predict_pathologies
        
        scan = np.random.rand(256, 256).astype(np.float32)
        
        # Test CR modality
        result_cr = predict_pathologies(scan, modality="CR")
        assert "primary_finding" in result_cr, "Primary finding missing"
        assert "primary_confidence" in result_cr, "Confidence missing"
        assert result_cr["primary_confidence"] > 0, "Confidence should be positive"
        print(f"✓ CR Classification: {result_cr['primary_finding']} ({result_cr['primary_confidence']:.3f})")
        
        # Test CT modality
        result_ct = predict_pathologies(scan, modality="CT")
        print(f"✓ CT Classification: {result_ct['primary_finding']} ({result_ct['primary_confidence']:.3f})")
        
        # Test MR modality
        result_mr = predict_pathologies(scan, modality="MR")
        print(f"✓ MR Classification: {result_mr['primary_finding']} ({result_mr['primary_confidence']:.3f})")
        
        print("\n✓ Classification tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Classification failed: {e}\n")
        return False

def test_segmentation():
    """Test 3D U-Net segmentation."""
    print("=" * 70)
    print("TEST 5: Volumetric Segmentation")
    print("=" * 70)
    
    try:
        import numpy as np
        from models.segmenter import generate_segmentation_mask, compute_dice_iou
        
        scan = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
        
        # Test segmentation
        result = generate_segmentation_mask(scan, roi_center_norm=(0.3, 0.6))
        assert result["mask"].shape == (256, 256), "Mask shape mismatch"
        assert result["ActivePixels"] > 0, "No active pixels"
        assert result["EstimatedVolumeCm3"] > 0, "Volume should be positive"
        print(f"✓ Segmentation mask: {result['ActivePixels']} pixels")
        print(f"✓ Lesion area: {result['AreaMm2']:.2f} mm²")
        print(f"✓ Estimated volume: {result['EstimatedVolumeCm3']:.2f} cm³")
        
        # Test Dice/IoU
        metrics = compute_dice_iou(result["mask"], result["mask"])
        assert abs(metrics["dice"] - 1.0) < 0.01, "Self-Dice should be ~1.0"
        assert abs(metrics["iou"] - 1.0) < 0.01, "Self-IoU should be ~1.0"
        print(f"✓ Dice coefficient: {metrics['dice']:.4f}")
        print(f"✓ IoU: {metrics['iou']:.4f}")
        
        print("\n✓ Segmentation tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Segmentation failed: {e}\n")
        return False

def test_gradcam():
    """Test Grad-CAM explainability."""
    print("=" * 70)
    print("TEST 6: Grad-CAM Explainability")
    print("=" * 70)
    
    try:
        import numpy as np
        from explainability.gradcam import GradCAMExplainer
        
        explainer = GradCAMExplainer()
        scan = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
        
        result = explainer.compute_heatmap(scan, focus_center_norm=(0.3, 0.6))
        assert result["heatmap_norm"].shape == (256, 256), "Heatmap shape mismatch"
        assert result["heatmap_rgb"].shape == (256, 256, 3), "RGB heatmap shape mismatch"
        assert 0 <= result["max_activation"] <= 1.0, "Activation out of range"
        print(f"✓ Heatmap generated: shape={result['heatmap_norm'].shape}")
        print(f"✓ Peak coordinates: {result['peak_coordinates_xy']}")
        print(f"✓ Mean activation: {result['mean_activation']:.4f}")
        print(f"✓ Max activation: {result['max_activation']:.4f}")
        
        print("\n✓ Grad-CAM tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Grad-CAM failed: {e}\n")
        return False

def test_orthanc_service():
    """Test Orthanc integration service."""
    print("=" * 70)
    print("TEST 7: Orthanc Study Ingestion")
    print("=" * 70)
    
    try:
        import numpy as np
        from orthanc_service import OrthancStudyIngestor
        
        ingestor = OrthancStudyIngestor(orthanc_url="http://localhost:8042")
        scan = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
        
        study = ingestor.process_dicom_study(
            study_id="TEST-STUDY-001",
            dicom_headers={"PatientID": "PT-999", "Modality": "CR", "WindowCenter": 500, "WindowWidth": 2000},
            pixel_array=scan,
        )
        
        assert study["study_id"] == "TEST-STUDY-001", "Study ID mismatch"
        assert "classification" in study, "Classification missing"
        assert "segmentation" in study, "Segmentation missing"
        assert "gradcam" in study, "Grad-CAM missing"
        assert study["review_status"] == "PENDING_RADIOLOGIST_REVIEW", "Status incorrect"
        print(f"✓ Study processed: {study['study_id']}")
        print(f"✓ Inference latency: {study['inference_latency_ms']:.2f} ms")
        print(f"✓ Classification: {study['classification']['primary_finding']}")
        print(f"✓ Segmentation volume: {study['segmentation']['estimated_volume_cm3']:.2f} cm³")
        
        print("\n✓ Orthanc service tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Orthanc service failed: {e}\n")
        return False

def test_fastapi_backend():
    """Test FastAPI backend and audit logging."""
    print("=" * 70)
    print("TEST 8: FastAPI Backend & Audit Log")
    print("=" * 70)
    
    try:
        from main import app, audit_logger, RadiologistDecisionRequest
        
        # Test audit logger
        req = RadiologistDecisionRequest(
            study_id="TEST-001",
            patient_id="PT-999",
            modality="CR",
            ai_finding="Pneumonia",
            ai_confidence=0.85,
            decision="APPROVED",
            radiologist_name="Dr. Test Radiologist"
        )
        
        result = audit_logger.record_decision(req)
        assert result["audit_id"] > 0, "Audit ID missing"
        assert result["decision"] == "APPROVED", "Decision mismatch"
        print(f"✓ Audit record created: ID={result['audit_id']}")
        print(f"✓ Timestamp: {result['timestamp_utc']}")
        
        # Test log retrieval
        logs = audit_logger.list_logs()
        assert len(logs) > 0, "No logs found"
        print(f"✓ Retrieved {len(logs)} audit log(s)")
        
        # Test rejection
        req_reject = RadiologistDecisionRequest(
            study_id="TEST-002",
            patient_id="PT-998",
            modality="CR",
            ai_finding="Cardiomegaly",
            ai_confidence=0.65,
            decision="REJECTED",
            radiologist_name="Dr. Test Radiologist",
            radiologist_notes="False positive due to ECG leads"
        )
        result_reject = audit_logger.record_decision(req_reject)
        assert result_reject["decision"] == "REJECTED", "Rejection not recorded"
        print(f"✓ Rejection recorded: ID={result_reject['audit_id']}")
        
        print("\n✓ FastAPI backend tests passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ FastAPI backend failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AIML-02 MEDICAL IMAGING SYSTEM - FUNCTIONALITY VERIFICATION")
    print("=" * 70 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("DICOM Processing", test_dicom_processing),
        ("Image Preprocessing", test_preprocessing),
        ("Pathology Classification", test_classification),
        ("Volumetric Segmentation", test_segmentation),
        ("Grad-CAM Explainability", test_gradcam),
        ("Orthanc Integration", test_orthanc_service),
        ("FastAPI Backend", test_fastapi_backend),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)\n")
    
    if passed == total:
        print("✓ ALL FUNCTIONALITY VERIFIED - SYSTEM READY FOR DEPLOYMENT")
        return 0
    else:
        print("✗ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        return 1

if __name__ == "__main__":
    sys.exit(main())
