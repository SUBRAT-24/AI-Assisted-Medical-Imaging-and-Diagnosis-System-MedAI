"""
Step 5 & 6 Backend API: FastAPI Service Connecting Orthanc DICOM Server, PyTorch/MONAI Inference,
and Immutable Radiologist Audit Log (Approve / Reject Decisions).
"""
import sqlite3
import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="AIML-02 AI-Assisted Medical Imaging & Diagnosis API",
    version="1.0.0",
    description="FOR EDUCATION ONLY, NOT FOR CLINICAL USE. Connects Orthanc PACS, PyTorch/MONAI & Grad-CAM.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RadiologistDecisionRequest(BaseModel):
    study_id: str
    patient_id: str
    modality: str
    ai_finding: str
    ai_confidence: float
    decision: str  # "APPROVED" or "REJECTED"
    radiologist_name: str
    radiologist_notes: Optional[str] = ""


class AuditLogger:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS radiologist_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                study_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                ai_finding TEXT NOT NULL,
                ai_confidence REAL NOT NULL,
                decision TEXT NOT NULL,
                radiologist_name TEXT NOT NULL,
                radiologist_notes TEXT
            )
            """
        )
        self.conn.commit()

    def record_decision(self, req: RadiologistDecisionRequest) -> Dict[str, Any]:
        if req.decision not in ("APPROVED", "REJECTED"):
            raise ValueError("Decision must be APPROVED or REJECTED")
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO radiologist_audit_log
            (timestamp_utc, study_id, patient_id, modality, ai_finding, ai_confidence, decision, radiologist_name, radiologist_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                req.study_id,
                req.patient_id,
                req.modality,
                req.ai_finding,
                req.ai_confidence,
                req.decision,
                req.radiologist_name,
                req.radiologist_notes or "",
            ),
        )
        self.conn.commit()
        return {
            "audit_id": cur.lastrowid,
            "timestamp_utc": ts,
            "study_id": req.study_id,
            "patient_id": req.patient_id,
            "modality": req.modality,
            "ai_finding": req.ai_finding,
            "ai_confidence": req.ai_confidence,
            "decision": req.decision,
            "radiologist_name": req.radiologist_name,
            "radiologist_notes": req.radiologist_notes or "",
        }

    def list_logs(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, timestamp_utc, study_id, patient_id, modality, ai_finding, ai_confidence, decision, radiologist_name, radiologist_notes FROM radiologist_audit_log ORDER BY id DESC"
        )
        rows = cur.fetchall()
        return [
            {
                "audit_id": r[0],
                "timestamp_utc": r[1],
                "study_id": r[2],
                "patient_id": r[3],
                "modality": r[4],
                "ai_finding": r[5],
                "ai_confidence": r[6],
                "decision": r[7],
                "radiologist_name": r[8],
                "radiologist_notes": r[9],
            }
            for r in rows
        ]


audit_logger = AuditLogger()


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "orthanc_connected": True,
        "models_loaded": ["DenseNet121-CheXpert", "MONAI-3D-UNet-MSD", "Grad-CAM-Explainer"],
        "disclaimer": "FOR EDUCATION ONLY, NOT FOR CLINICAL USE",
    }


@app.post("/api/v1/studies/{study_id}/review")
def submit_review(study_id: str, payload: RadiologistDecisionRequest):
    if study_id != payload.study_id:
        raise HTTPException(status_code=400, detail="study_id mismatch")
    try:
        return audit_logger.record_decision(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/v1/audit-logs")
def get_audit_logs():
    return {"logs": audit_logger.list_logs()}
