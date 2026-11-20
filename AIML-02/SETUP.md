# Setup and Testing Guide

## Prerequisites

- Python 3.11+ (currently using Python 3.13.5)
- Docker and Docker Compose
- Node.js 20+ (for frontend)
- Git

## Installation Steps

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Verify Installation

Run the automated test suite:

```bash
# From project root
python backend/tests/test_pipeline.py
```

## Running the Application

### Option A: Docker Compose (Recommended)

```bash
# From project root
docker compose up --build
```

Access points:
- **Frontend (React Viewer)**: http://localhost:3000
- **Backend API (FastAPI)**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Orthanc PACS**: http://localhost:8042

### Option B: Development Mode (Manual)

**Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Orthanc (separate terminal):**
```bash
docker run -p 4242:4242 -p 8042:8042 jodogne/orthanc-plugins:1.12.4
```

## Testing

### Run All Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Individual Components

**1. DICOM Processing:**
```bash
python -c "
from backend.dicom_utils import apply_hu_rescaling, apply_window_level
import numpy as np
scan = np.random.randint(0, 2000, size=(256, 256), dtype=np.int16)
hu = apply_hu_rescaling(scan, slope=1.0, intercept=-1024.0)
windowed = apply_window_level(hu, window_width=1500.0, window_center=-600.0)
print(f'HU conversion OK: min={hu.min()}, max={hu.max()}')
print(f'Windowing OK: shape={windowed.shape}, dtype={windowed.dtype}')
"
```

**2. Classification:**
```bash
python -c "
from backend.models.classifier import predict_pathologies
import numpy as np
scan = np.random.rand(256, 256).astype(np.float32)
result = predict_pathologies(scan, modality='CR')
print(f'Primary finding: {result[\"primary_finding\"]}')
print(f'Confidence: {result[\"primary_confidence\"]}')
"
```

**3. Segmentation:**
```bash
python -c "
from backend.models.segmenter import generate_segmentation_mask
import numpy as np
scan = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
result = generate_segmentation_mask(scan)
print(f'Active pixels: {result[\"ActivePixels\"]}')
print(f'Estimated volume: {result[\"EstimatedVolumeCm3\"]} cm³')
"
```

**4. Grad-CAM:**
```bash
python -c "
from backend.explainability.gradcam import GradCAMExplainer
import numpy as np
explainer = GradCAMExplainer()
scan = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
result = explainer.compute_heatmap(scan)
print(f'Heatmap shape: {result[\"heatmap_norm\"].shape}')
print(f'Peak activation: {result[\"max_activation\"]}')
"
```

**5. FastAPI Backend:**
```bash
python -c "
from backend.main import app, audit_logger, RadiologistDecisionRequest

# Test audit logger
req = RadiologistDecisionRequest(
    study_id='TEST-001',
    patient_id='PT-999',
    modality='CR',
    ai_finding='Pneumonia',
    ai_confidence=0.85,
    decision='APPROVED',
    radiologist_name='Dr. Test'
)
result = audit_logger.record_decision(req)
print(f'Audit record created: {result[\"audit_id\"]}')

# List logs
logs = audit_logger.list_logs()
print(f'Total logs: {len(logs)}')
"
```

## Functionality Checklist

- [x] DICOM metadata extraction and PHI anonymization
- [x] Hounsfield Unit rescaling and Window/Level transformation
- [x] 2D/3D image preprocessing with MONAI transforms
- [x] Multi-label pathology classification (DenseNet-121/EfficientNet-B4)
- [x] 3D volumetric segmentation (MONAI U-Net)
- [x] Grad-CAM explainability heatmaps
- [x] Orthanc DICOM server integration
- [x] FastAPI backend with audit logging
- [x] React frontend with diagnostic viewer
- [x] Docker Compose orchestration
- [x] Automated test suite

## Troubleshooting

### Issue: Dependencies not installed
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Issue: Docker containers fail to start
```bash
docker compose down -v
docker compose up --build --force-recreate
```

### Issue: Frontend build fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Issue: Port conflicts
Edit `docker-compose.yml` to use different ports:
- Change `3000:80` to `3001:80` for frontend
- Change `8000:8000` to `8001:8000` for backend
- Change `8042:8042` to `8043:8042` for Orthanc

## System Requirements

- **RAM**: Minimum 4GB (8GB recommended for Docker)
- **Storage**: 2GB for application, additional space for DICOM data
- **OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)

## Next Steps After Setup

1. Upload sample DICOM files to Orthanc via http://localhost:8042
2. View studies in the React diagnostic workstation at http://localhost:3000
3. Test AI inference and Grad-CAM visualization
4. Submit Approve/Reject decisions and view audit logs
5. Export audit logs as CSV for compliance review
