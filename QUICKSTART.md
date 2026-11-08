# Quick Start Guide - AIML-02 Medical Imaging System

## 🚀 Get Started in 5 Minutes

### Prerequisites Check
```bash
python --version   # Should be 3.11+
docker --version   # Should be 20.10+
node --version     # Should be 20+
git --version      # Any recent version
```

### 1. Clone Repository
```bash
git clone https://github.com/SUBRAT-24/AI-Assisted-Medical-Imaging-and-Diagnosis-System-MedAI.git
cd AI-Assisted-Medical-Imaging-and-Diagnosis-System-MedAI
```

### 2. Start with Docker (Recommended)
```bash
docker compose up --build -d
```

Wait 2-3 minutes for all services to start, then access:

- **🖥️ React Viewer**: http://localhost:3000
- **🔌 API Docs**: http://localhost:8000/docs
- **📡 Orthanc PACS**: http://localhost:8042

### 3. Quick Test

**Option A: Use Pre-loaded Sample Studies**
- Open http://localhost:3000
- Three sample studies are pre-loaded:
  1. Chest X-Ray (Pneumonia detection)
  2. Brain MRI (Glioma segmentation)
  3. Thoracic CT (Lung nodule detection)

**Option B: Verify Backend**
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "orthanc_connected": true,
  "models_loaded": [
    "DenseNet121-CheXpert",
    "MONAI-3D-UNet-MSD",
    "Grad-CAM-Explainer"
  ]
}
```

## 📝 Basic Usage

### View Medical Scans
1. Open http://localhost:3000
2. Select a study from the dropdown
3. Adjust Window/Level sliders to change contrast
4. Toggle Grad-CAM and segmentation overlays
5. Use 3D slice scrubber for volumetric data

### Submit Radiologist Review
1. Enter radiologist name
2. Select reason category
3. Add clinical notes (optional)
4. Click **Approve** or **Reject**
5. View audit log in the "Audit Trail" tab

### View Audit Logs
1. Click "Audit Trail" tab
2. Filter by decision type (All/Approved/Rejected)
3. Search by study ID, patient ID, or finding
4. Export logs as CSV for compliance

## 🛠️ Development Mode

### Backend Only
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend Only
```bash
cd frontend
npm install
npm run dev
```

### Run Tests
```bash
# Python backend tests
python verify_functionality.py

# Unit tests
python -m unittest discover -s tests -p "test_*.py" -v
```

## 🔧 Configuration

### Change Ports
Edit `docker-compose.yml`:
```yaml
services:
  react-radiology-viewer:
    ports:
      - "3001:80"  # Change 3000 to 3001
  
  ai-inference-backend:
    ports:
      - "8001:8000"  # Change 8000 to 8001
```

### Environment Variables
Create `.env` file:
```bash
# Backend
ORTHANC_URL=http://orthanc-pacs:8042
MODEL_DEVICE=cpu  # or 'cuda' for GPU
DATABASE_URL=sqlite:///./data/audit_log.db

# Frontend
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_ORTHANC_URL=http://localhost:8042
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  REACT DIAGNOSTIC VIEWER                    │
│              (Port 3000 - Web Interface)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────┴───────────────────────────────────────┐
│            FASTAPI BACKEND (Port 8000)                      │
│  • Classification (DenseNet-121)                            │
│  • Segmentation (MONAI 3D U-Net)                            │
│  • Grad-CAM Explainability                                  │
│  • Audit Logging                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ DICOM REST API
┌─────────────────────┴───────────────────────────────────────┐
│         ORTHANC PACS SERVER (Ports 4242, 8042)              │
│  • DICOM Storage                                            │
│  • Study Management                                         │
└─────────────────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Services won't start
```bash
docker compose down -v
docker compose up --build --force-recreate
```

### Port already in use
```bash
# Find process using port
netstat -ano | findstr :3000

# Kill process (Windows)
taskkill /PID <PID> /F
```

### Frontend shows "API connection failed"
- Check backend is running: `curl http://localhost:8000/api/v1/health`
- Check browser console for CORS errors
- Verify API URL in frontend environment variables

### Can't access Orthanc
- Wait 30 seconds after `docker compose up`
- Check: `curl http://localhost:8042/system`
- Check logs: `docker compose logs orthanc-pacs`

## 📚 Next Steps

- Read [SETUP.md](SETUP.md) for detailed installation
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Review [README.md](README.md) for architecture details
- Check [tests/test_pipeline.py](tests/test_pipeline.py) for test examples

## ⚠️ Important Notice

**FOR EDUCATIONAL AND RESEARCH USE ONLY**

This system is NOT approved for clinical diagnostic use. All AI predictions require verification by qualified radiologists. See README.md for full disclaimer.

## 🆘 Support

- **Issues**: https://github.com/SUBRAT-24/AI-Assisted-Medical-Imaging-and-Diagnosis-System-MedAI/issues
- **Documentation**: All .md files in repository root
- **API Docs**: http://localhost:8000/docs (when running)

## ✅ Verification Checklist

After setup, verify these work:

- [ ] Backend health check returns 200 OK
- [ ] Frontend loads at localhost:3000
- [ ] Can view pre-loaded sample studies
- [ ] Grad-CAM overlay toggles on/off
- [ ] Segmentation mask displays correctly
- [ ] Window/Level sliders adjust image
- [ ] Can submit Approve/Reject decisions
- [ ] Audit logs appear in Audit Trail tab
- [ ] Can export audit logs as CSV
- [ ] Orthanc web UI accessible at localhost:8042

---

**Ready to start?** Run `docker compose up --build -d` and open http://localhost:3000 🎉
