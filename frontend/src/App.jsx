import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Activity,
  ShieldAlert,
  CheckCircle2,
  XCircle,
  Upload,
  RefreshCw,
  Layers,
  Eye,
  Sliders,
  FileText,
  Database,
  Server,
  Cpu,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  FlipHorizontal,
  Contrast,
  Download,
  Copy,
  Check,
  Search,
  AlertTriangle,
  Brain,
  Stethoscope,
  ClipboardCheck,
  BarChart3,
  BookOpen,
  Terminal,
  Sparkles,
  Crosshair,
  ChevronRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line
} from 'recharts';

// Embedded clinical scan references (inlined as base64 data URIs during build)
const SCAN_IMAGES = {
  chest_xray: "user_uploaded_files/ttl=63d/tool_generated/chest_xray.jpg",
  brain_mri: "user_uploaded_files/ttl=63d/tool_generated/brain_mri.jpg",
  lung_ct: "user_uploaded_files/ttl=63d/tool_generated/lung_ct.jpg"
};

const INITIAL_STUDIES = [
  {
    id: "ORTHANC-CXR-1042",
    orthancUuid: "8f31a9c0-41b2-4d8e-9a12-09c41f77e104",
    patientId: "ANON-CXR-8821",
    patientAgeSex: "58Y / M",
    studyDate: "2026-10-30 09:14 UTC",
    modality: "CR",
    modalityName: "Chest X-Ray (PA Radiograph)",
    datasetSource: "NIH ChestX-ray14 / CheXpert v1.0",
    imageSrc: SCAN_IMAGES.chest_xray,
    is3D: false,
    totalSlices: 1,
    defaultSlice: 1,
    pixelSpacing: [0.143, 0.143],
    sliceThickness: 1.0,
    defaultWW: 2000,
    defaultWL: 500,
    roiCenter: { x: 0.28, y: 0.63 },
    roiRadii: { rx: 0.14, ry: 0.13 },
    secondaryRoi: { x: 0.33, y: 0.56, rx: 0.08, ry: 0.08 },
    modelBackbone: "DenseNet-121 (Transfer Learning, 14-Class CheXpert)",
    segArchitecture: "MONAI Attention U-Net ROI Mask",
    primaryFinding: "Right Lower Lobe Pneumonia / Consolidation",
    primaryConfidence: 0.914,
    lesionAreaMm2: 1842.6,
    lesionVolumeCm3: 14.8,
    diceScoreVal: 0.864,
    inferenceTimeMs: 142,
    status: "PENDING",
    dicomTags: [
      { tag: "(0008,0016)", name: "SOPClassUID", value: "1.2.840.10008.5.1.4.1.1.1 (Computed Radiography)" },
      { tag: "(0008,0060)", name: "Modality", value: "CR" },
      { tag: "(0010,0010)", name: "PatientName", value: "ANONYMIZED (HIPAA Safe Harbor)" },
      { tag: "(0010,0020)", name: "PatientID", value: "ANON-CXR-8821" },
      { tag: "(0018,0015)", name: "BodyPartExamined", value: "CHEST" },
      { tag: "(0018,5101)", name: "ViewPosition", value: "PA" },
      { tag: "(0028,0030)", name: "PixelSpacing", value: "0.143 \\ 0.143 mm" },
      { tag: "(0028,1050)", name: "WindowCenter", value: "500" },
      { tag: "(0028,1051)", name: "WindowWidth", value: "2000" }
    ],
    predictions: [
      { finding: "Pneumonia", confidence: 0.914, threshold: 0.5 },
      { finding: "Consolidation", confidence: 0.882, threshold: 0.5 },
      { finding: "Infiltration", confidence: 0.746, threshold: 0.5 },
      { finding: "Pleural Effusion", confidence: 0.312, threshold: 0.5 },
      { finding: "Atelectasis", confidence: 0.264, threshold: 0.5 },
      { finding: "Cardiomegaly", confidence: 0.118, threshold: 0.5 },
      { finding: "Pneumothorax", confidence: 0.024, threshold: 0.5 }
    ],
    clinicalSummary: "Focal airspace opacity in the right lower lung zone with air bronchograms, highly concordant with lobar consolidation/pneumonia. Left lung remains clear; cardiomediastinal silhouette within normal limits."
  },
  {
    id: "ORTHANC-MRI-2089",
    orthancUuid: "c492e110-77a9-4e11-8f22-51b89021a208",
    patientId: "ANON-BRATS-0419",
    patientAgeSex: "46Y / F",
    studyDate: "2026-10-30 11:40 UTC",
    modality: "MR",
    modalityName: "3D Brain MRI (Axial T2-FLAIR)",
    datasetSource: "Medical Segmentation Decathlon Task01_BrainTumour",
    imageSrc: SCAN_IMAGES.brain_mri,
    is3D: true,
    totalSlices: 64,
    defaultSlice: 34,
    pixelSpacing: [1.0, 1.0],
    sliceThickness: 1.5,
    defaultWW: 220,
    defaultWL: 90,
    roiCenter: { x: 0.25, y: 0.62 },
    roiRadii: { rx: 0.10, ry: 0.12 },
    secondaryRoi: { x: 0.27, y: 0.62, rx: 0.14, ry: 0.16 },
    modelBackbone: "EfficientNet-B4 + MONAI 3D Residual U-Net",
    segArchitecture: "MONAI 3D U-Net (5-Stage Residual, DiceCELoss)",
    primaryFinding: "Right Temporal-Parietal Hyperintense Glioma + Edema",
    primaryConfidence: 0.948,
    lesionAreaMm2: 1495.2,
    lesionVolumeCm3: 24.6,
    diceScoreVal: 0.894,
    inferenceTimeMs: 386,
    status: "PENDING",
    dicomTags: [
      { tag: "(0008,0016)", name: "SOPClassUID", value: "1.2.840.10008.5.1.4.1.1.4 (MR Image Storage)" },
      { tag: "(0008,0060)", name: "Modality", value: "MR" },
      { tag: "(0010,0010)", name: "PatientName", value: "ANONYMIZED (HIPAA Safe Harbor)" },
      { tag: "(0010,0020)", name: "PatientID", value: "ANON-BRATS-0419" },
      { tag: "(0018,0024)", name: "SequenceName", value: "T2_FLAIR_AXIAL_3D" },
      { tag: "(0018,0050)", name: "SliceThickness", value: "1.5 mm" },
      { tag: "(0018,0080)", name: "RepetitionTime (TR)", value: "9000.0 ms" },
      { tag: "(0018,0081)", name: "EchoTime (TE)", value: "125.0 ms" },
      { tag: "(0028,0030)", name: "PixelSpacing", value: "1.000 \\ 1.000 mm" }
    ],
    predictions: [
      { finding: "Hyperintense FLAIR Tumor Core", confidence: 0.948, threshold: 0.5 },
      { finding: "Vasogenic Peritumoral Edema", confidence: 0.912, threshold: 0.5 },
      { finding: "Local Sulcal Effacement / Mass Effect", confidence: 0.764, threshold: 0.5 },
      { finding: "Midline Shift (>3mm)", confidence: 0.185, threshold: 0.5 },
      { finding: "Intraventricular Extension", confidence: 0.092, threshold: 0.5 }
    ],
    clinicalSummary: "3D MONAI U-Net segments a heterogeneous hyperintense T2-FLAIR mass in the right temporoparietal cortex (radiological left side of image) measuring 24.6 cm³ with surrounding vasogenic edema and mild local sulcal effacement."
  },
  {
    id: "ORTHANC-CT-3115",
    orthancUuid: "a190b772-33f1-4c90-811b-77e01299c311",
    patientId: "ANON-LUNG-5504",
    patientAgeSex: "64Y / M",
    studyDate: "2026-10-30 13:05 UTC",
    modality: "CT",
    modalityName: "3D Thoracic CT (Lung Window)",
    datasetSource: "Medical Segmentation Decathlon Task06_Lung",
    imageSrc: SCAN_IMAGES.lung_ct,
    is3D: true,
    totalSlices: 64,
    defaultSlice: 28,
    pixelSpacing: [0.78, 0.78],
    sliceThickness: 1.25,
    defaultWW: 1500,
    defaultWL: -600,
    roiCenter: { x: 0.77, y: 0.40 },
    roiRadii: { rx: 0.045, ry: 0.045 },
    secondaryRoi: { x: 0.77, y: 0.40, rx: 0.075, ry: 0.075 },
    modelBackbone: "DenseNet-121 + MONAI 3D U-Net Sliding Window",
    segArchitecture: "MONAI 3D U-Net (Patch 128×128×64, Overlap 0.5)",
    primaryFinding: "Left Upper Lobe Spiculated Pulmonary Nodule (Lung-RADS 4B)",
    primaryConfidence: 0.926,
    lesionAreaMm2: 312.4,
    lesionVolumeCm3: 3.85,
    diceScoreVal: 0.838,
    inferenceTimeMs: 410,
    status: "PENDING",
    dicomTags: [
      { tag: "(0008,0016)", name: "SOPClassUID", value: "1.2.840.10008.5.1.4.1.1.2 (CT Image Storage)" },
      { tag: "(0008,0060)", name: "Modality", value: "CT" },
      { tag: "(0010,0010)", name: "PatientName", value: "ANONYMIZED (HIPAA Safe Harbor)" },
      { tag: "(0010,0020)", name: "PatientID", value: "ANON-LUNG-5504" },
      { tag: "(0018,0050)", name: "SliceThickness", value: "1.25 mm" },
      { tag: "(0018,0060)", name: "KVP", value: "120 kV" },
      { tag: "(0028,1050)", name: "WindowCenter", value: "-600 HU" },
      { tag: "(0028,1051)", name: "WindowWidth", value: "1500 HU" },
      { tag: "(0028,1052)", name: "RescaleIntercept", value: "-1024.0" },
      { tag: "(0028,1053)", name: "RescaleSlope", value: "1.0" }
    ],
    predictions: [
      { finding: "Pulmonary Nodule (LUL)", confidence: 0.926, threshold: 0.5 },
      { finding: "Spiculated Margin / Pleural Tag", confidence: 0.854, threshold: 0.5 },
      { finding: "Centrilobular Emphysema", confidence: 0.284, threshold: 0.5 },
      { finding: "Mediastinal Lymphadenopathy", confidence: 0.142, threshold: 0.5 },
      { finding: "Pleural Effusion", confidence: 0.041, threshold: 0.5 }
    ],
    clinicalSummary: "Well-circumscribed 18.5 mm solid pulmonary nodule in the left upper lobe (radiological right side of axial CT image) segmented by MONAI 3D U-Net (volume 3.85 cm³). Recommend PET-CT or tissue biopsy per Fleischner / Lung-RADS 4B guidelines."
  }
];

const INITIAL_AUDIT_LOGS = [
  {
    auditId: "AUD-2026-0901",
    timestamp: "2026-10-30 08:42:19 UTC",
    studyId: "ORTHANC-CXR-0998",
    patientId: "ANON-CXR-7741",
    modality: "CR",
    aiFinding: "Cardiomegaly (CTR 0.58)",
    aiConfidence: 0.932,
    decision: "APPROVED",
    radiologist: "Dr. A. Sharma, MD (Thoracic Radiology)",
    reasonCategory: "True Positive — Concordant Grad-CAM & Anatomy",
    notes: "Grad-CAM accurately highlights bilateral cardiac borders; cardiothoracic ratio measured at 0.58 on PA projection.",
    sha256: "e91a4f7c...882b10"
  },
  {
    auditId: "AUD-2026-0902",
    timestamp: "2026-10-30 09:05:44 UTC",
    studyId: "ORTHANC-CXR-1015",
    patientId: "ANON-CXR-8102",
    modality: "CR",
    aiFinding: "Pneumothorax (Apical)",
    aiConfidence: 0.684,
    decision: "REJECTED",
    radiologist: "Dr. M. Nair, MD (Neuroradiology & Chest)",
    reasonCategory: "Shortcut Learning / Confounder Artifact",
    notes: "Rejected AI alert: Grad-CAM heatmap focused on an external ECG lead wire and skin fold mimicking a visceral pleural line.",
    sha256: "4b7c091e...319d4a"
  }
];

const BENCHMARK_AUROC_DATA = [
  { pathology: "Cardiomegaly", DenseNet121: 0.928, EfficientNetB4: 0.921 },
  { pathology: "Effusion", DenseNet121: 0.912, EfficientNetB4: 0.908 },
  { pathology: "Pneumothorax", DenseNet121: 0.895, EfficientNetB4: 0.889 },
  { pathology: "Pneumonia", DenseNet121: 0.887, EfficientNetB4: 0.884 },
  { pathology: "Edema", DenseNet121: 0.881, EfficientNetB4: 0.876 },
  { pathology: "Consolidation", DenseNet121: 0.874, EfficientNetB4: 0.869 },
  { pathology: "Mass", DenseNet121: 0.846, EfficientNetB4: 0.858 },
  { pathology: "Nodule", DenseNet121: 0.819, EfficientNetB4: 0.834 }
];

const UNET_TRAINING_HISTORY = [
  { epoch: 10, BrainDice: 0.64, LungDice: 0.58, ValLoss: 0.49 },
  { epoch: 25, BrainDice: 0.76, LungDice: 0.71, ValLoss: 0.34 },
  { epoch: 50, BrainDice: 0.83, LungDice: 0.78, ValLoss: 0.24 },
  { epoch: 75, BrainDice: 0.87, LungDice: 0.82, ValLoss: 0.19 },
  { epoch: 100, BrainDice: 0.894, LungDice: 0.838, ValLoss: 0.16 }
];

// Color LUT helper for Grad-CAM heatmaps
function getColormapRGB(value, scheme = "jet") {
  const v = Math.max(0, Math.min(1, value));
  if (scheme === "inferno") {
    const r = Math.min(255, Math.round(255 * Math.pow(v, 0.7) * 1.2));
    const g = Math.min(255, Math.round(255 * Math.pow(Math.max(0, v - 0.25) / 0.75, 1.4)));
    const b = Math.min(255, Math.round(255 * (v < 0.5 ? v * 1.4 : Math.max(0, (1 - v) * 1.8))));
    return [r, g, b];
  }
  if (scheme === "viridis") {
    const r = Math.round(255 * Math.max(0, Math.min(1, -0.2 + 1.4 * Math.pow(v, 1.5))));
    const g = Math.round(255 * Math.max(0, Math.min(1, 0.1 + 0.85 * v)));
    const b = Math.round(255 * Math.max(0, Math.min(1, 0.55 + 0.4 * Math.sin(v * Math.PI) - 0.5 * v)));
    return [r, g, b];
  }
  // Default: Clinical Jet LUT
  const fourValue = 4 * v;
  const r = Math.round(255 * Math.min(fourValue - 1.5, -fourValue + 4.5));
  const g = Math.round(255 * Math.min(fourValue - 0.5, -fourValue + 3.5));
  const b = Math.round(255 * Math.min(fourValue + 0.5, -fourValue + 2.5));
  return [
    Math.max(0, Math.min(255, r)),
    Math.max(0, Math.min(255, g)),
    Math.max(0, Math.min(255, b))
  ];
}

export default function App() {
  const [activeTab, setActiveTab] = useState("workstation");
  const [studies, setStudies] = useState(INITIAL_STUDIES);
  const [selectedStudyId, setSelectedStudyId] = useState(INITIAL_STUDIES[0].id);

  // Viewer Controls
  const [showGradCam, setShowGradCam] = useState(true);
  const [gradCamOpacity, setGradCamOpacity] = useState(0.58);
  const [gradCamColormap, setGradCamColormap] = useState("jet");
  const [showSegMask, setShowSegMask] = useState(true);
  const [showCrosshair, setShowCrosshair] = useState(true);
  const [invertGrayscale, setInvertGrayscale] = useState(false);
  const [flipH, setFlipH] = useState(false);
  const [zoom, setZoom] = useState(1.0);
  const [windowWidth, setWindowWidth] = useState(INITIAL_STUDIES[0].defaultWW);
  const [windowLevel, setWindowLevel] = useState(INITIAL_STUDIES[0].defaultWL);
  const [currentSlice, setCurrentSlice] = useState(INITIAL_STUDIES[0].defaultSlice);
  const [cursorProbe, setCursorProbe] = useState({ x: 144, y: 320, hu: -142, camActivation: 0.89 });

  // Radiologist Decision Form State
  const [radiologistName, setRadiologistName] = useState("Dr. A. Sharma, MD (Radiology)");
  const [reasonCategory, setReasonCategory] = useState("True Positive — Concordant Grad-CAM & Anatomy");
  const [radiologistNotes, setRadiologistNotes] = useState("");
  const [toastMessage, setToastMessage] = useState(null);

  // Orthanc Poller & DVC Preprocessing State
  const [isPollingOrthanc, setIsPollingOrthanc] = useState(false);
  const [pollerStepText, setPollerStepText] = useState("");
  const [prepConfig, setPrepConfig] = useState({
    resize224: true,
    zScoreNorm: true,
    randRotate: true,
    randFlip: true,
    gaussianNoise: false
  });

  // Audit Log State (persisted safely in localStorage)
  const [auditLogs, setAuditLogs] = useState(() => {
    try {
      const saved = localStorage.getItem("aiml02_audit_logs_v1");
      return saved ? JSON.parse(saved) : INITIAL_AUDIT_LOGS;
    } catch {
      return INITIAL_AUDIT_LOGS;
    }
  });
  const [auditSearch, setAuditSearch] = useState("");
  const [auditFilterDecision, setAuditFilterDecision] = useState("ALL");
  const [copiedAuditJson, setCopiedAuditJson] = useState(false);

  const canvasRef = useRef(null);
  const loadedImageRef = useRef(null);
  const fileInputRef = useRef(null);

  const activeStudy = useMemo(
    () => studies.find((s) => s.id === selectedStudyId) || studies[0],
    [studies, selectedStudyId]
  );

  const showToast = useCallback((msg, type = "info") => {
    setToastMessage({ msg, type });
    setTimeout(() => {
      setToastMessage((prev) => (prev && prev.msg === msg ? null : prev));
    }, 4000);
  }, []);

  // Sync default WW/WL and 3D slice when switching studies
  useEffect(() => {
    if (!activeStudy) return;
    setWindowWidth(activeStudy.defaultWW);
    setWindowLevel(activeStudy.defaultWL);
    setCurrentSlice(activeStudy.defaultSlice);
    setZoom(1.0);
    setInvertGrayscale(false);
    setFlipH(false);
  }, [activeStudy.id]);

  // Persist Audit Logs safely
  useEffect(() => {
    try {
      localStorage.setItem("aiml02_audit_logs_v1", JSON.stringify(auditLogs));
    } catch {
      // Ignore storage quota errors in sandbox
    }
  }, [auditLogs]);

  // Render medical scan + WW/WL transformation + 3D slice depth modulation + Grad-CAM + MONAI 3D U-Net overlay on HTML5 Canvas
  const renderDiagnosticCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = loadedImageRef.current;
    if (!canvas || !img || !activeStudy) return;

    const ctx = canvas.getContext("2d");
    const width = 512;
    const height = 512;
    canvas.width = width;
    canvas.height = height;

    // Offscreen canvas to read source pixel intensities
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");

    offCtx.save();
    if (flipH) {
      offCtx.translate(width, 0);
      offCtx.scale(-1, 1);
    }
    offCtx.drawImage(img, 0, 0, width, height);
    offCtx.restore();

    const imgData = offCtx.getImageData(0, 0, width, height);
    const data = imgData.data;

    // Map Window Width / Window Level to contrast/brightness transfer function
    const wwRatio = Math.max(0.15, windowWidth / activeStudy.defaultWW);
    const wlDelta = ((windowLevel - activeStudy.defaultWL) / Math.max(100, activeStudy.defaultWW)) * 180;

    // 3D slice modulation factor (for CT/MRI volumetric scrubbing)
    const sliceOffset = activeStudy.is3D
      ? (currentSlice - activeStudy.defaultSlice) / (activeStudy.totalSlices * 0.5)
      : 0;
    const sliceScale = Math.max(0.35, 1 - Math.abs(sliceOffset) * 1.15);

    const cx = (flipH ? 1 - activeStudy.roiCenter.x : activeStudy.roiCenter.x) * width;
    const cy = activeStudy.roiCenter.y * height + sliceOffset * 18;
    const rx = activeStudy.roiRadii.rx * width * sliceScale;
    const ry = activeStudy.roiRadii.ry * height * sliceScale;

    const sx2 = (flipH ? 1 - activeStudy.secondaryRoi.x : activeStudy.secondaryRoi.x) * width;
    const sy2 = activeStudy.secondaryRoi.y * height + sliceOffset * 14;
    const srx2 = activeStudy.secondaryRoi.rx * width * sliceScale;
    const sry2 = activeStudy.secondaryRoi.ry * height * sliceScale;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const rawLum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];

        // Apply WW/WL contrast & brightness curve
        let windowed = (rawLum - 128 - wlDelta) / wwRatio + 128;
        windowed = Math.max(0, Math.min(255, windowed));
        if (invertGrayscale) {
          windowed = 255 - windowed;
        }

        let rOut = windowed;
        let gOut = windowed;
        let bOut = windowed;

        // Compute spatial Grad-CAM activation (combines anatomical focus kernel + local image intensity)
        if (showGradCam && gradCamOpacity > 0.01) {
          const d1 = ((x - cx) * (x - cx)) / (2 * rx * rx + 1) + ((y - cy) * (y - cy)) / (2 * ry * ry + 1);
          const d2 = ((x - sx2) * (x - sx2)) / (2 * srx2 * srx2 + 1) + ((y - sy2) * (y - sy2)) / (2 * sry2 * sry2 + 1);
          const g1 = Math.exp(-d1 * 0.85);
          const g2 = 0.55 * Math.exp(-d2 * 0.9);
          const intensityFactor = 0.45 + 0.55 * (rawLum / 255.0);
          const camVal = Math.min(1.0, Math.max(g1, g2) * intensityFactor * sliceScale);

          if (camVal > 0.08) {
            const [cr, cg, cb] = getColormapRGB(camVal, gradCamColormap);
            const alpha = gradCamOpacity * Math.min(1, camVal * 1.35);
            rOut = Math.round((1 - alpha) * rOut + alpha * cr);
            gOut = Math.round((1 - alpha) * gOut + alpha * cg);
            bOut = Math.round((1 - alpha) * bOut + alpha * cb);
          }
        }

        data[idx] = rOut;
        data[idx + 1] = gOut;
        data[idx + 2] = bOut;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Draw MONAI 3D U-Net Segmentation Mask & Contours
    if (showSegMask && sliceScale > 0.38) {
      ctx.save();
      // Outer Perilesional Edema / Margin Zone (Cyan dashed contour)
      ctx.beginPath();
      ctx.ellipse(sx2, sy2, srx2 * 1.15, sry2 * 1.15, 0.2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(6, 182, 212, 0.14)";
      ctx.fill();
      ctx.lineWidth = 1.75;
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#22d3ee";
      ctx.stroke();

      // Primary Core Segmentation Mask (Emerald/Lime solid contour)
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0.15, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(16, 185, 129, 0.22)";
      ctx.fill();
      ctx.setLineDash([]);
      ctx.lineWidth = 2.25;
      ctx.strokeStyle = "#10b981";
      ctx.stroke();

      // Caliper measurement line across primary segmented ROI
      ctx.beginPath();
      ctx.moveTo(cx - rx, cy);
      ctx.lineTo(cx + rx, cy);
      ctx.strokeStyle = "rgba(250, 204, 21, 0.85)";
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.stroke();

      // ROI Badge Label
      ctx.setLineDash([]);
      const badgeText = `3D U-Net ROI • ${(activeStudy.lesionVolumeCm3 * sliceScale).toFixed(1)} cm³`;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      const textWidth = ctx.measureText(badgeText).width;
      const bx = Math.max(10, Math.min(width - textWidth - 20, cx - textWidth / 2));
      const by = Math.max(24, cy - ry - 12);

      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      ctx.fillRect(bx - 6, by - 13, textWidth + 12, 19);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1;
      ctx.strokeRect(bx - 6, by - 13, textWidth + 12, 19);
      ctx.fillStyle = "#ecfdf5";
      ctx.fillText(badgeText, bx, by);
      ctx.restore();
    }

    // Draw PACS Crosshair & Corner Overlays
    if (showCrosshair) {
      ctx.save();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.28)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursorProbe.x, 0);
      ctx.lineTo(cursorProbe.x, height);
      ctx.moveTo(0, cursorProbe.y);
      ctx.lineTo(width, cursorProbe.y);
      ctx.stroke();
      ctx.restore();
    }
  }, [
    activeStudy,
    windowWidth,
    windowLevel,
    currentSlice,
    invertGrayscale,
    flipH,
    showGradCam,
    gradCamOpacity,
    gradCamColormap,
    showSegMask,
    showCrosshair,
    cursorProbe.x,
    cursorProbe.y
  ]);

  // Load image whenever activeStudy changes
  useEffect(() => {
    if (!activeStudy) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      loadedImageRef.current = img;
      renderDiagnosticCanvas();
    };
    img.src = activeStudy.imageSrc;
  }, [activeStudy, renderDiagnosticCanvas]);

  // Re-render canvas when controls change
  useEffect(() => {
    renderDiagnosticCanvas();
  }, [renderDiagnosticCanvas]);

  // Handle mouse movement on diagnostic canvas to compute live HU & Grad-CAM activation readout
  const handleCanvasMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !activeStudy) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(511, Math.round(((e.clientX - rect.left) / rect.width) * 512)));
    const y = Math.max(0, Math.min(511, Math.round(((e.clientY - rect.top) / rect.height) * 512)));

    const cx = (flipH ? 1 - activeStudy.roiCenter.x : activeStudy.roiCenter.x) * 512;
    const cy = activeStudy.roiCenter.y * 512;
    const rx = activeStudy.roiRadii.rx * 512;
    const ry = activeStudy.roiRadii.ry * 512;
    const dist = ((x - cx) * (x - cx)) / (2 * rx * rx + 1) + ((y - cy) * (y - cy)) / (2 * ry * ry + 1);
    const camAct = Math.min(0.99, Math.max(0.02, Math.exp(-dist * 0.75)));
    const simulatedHu =
      activeStudy.modality === "CT"
        ? Math.round(-750 + camAct * 810)
        : Math.round(windowLevel + (camAct - 0.5) * 320);

    setCursorProbe({ x, y, hu: simulatedHu, camActivation: Number(camAct.toFixed(2)) });
  };

  // Handle Radiologist Approve / Reject Decision
  const handleRadiologistDecision = (decisionType) => {
    if (!activeStudy) return;
    const nowUtc = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
    const newAuditId = `AUD-2026-${String(903 + auditLogs.length).padStart(4, "0")}`;
    const hashSuffix = Math.random().toString(16).slice(2, 8);

    const newEntry = {
      auditId: newAuditId,
      timestamp: nowUtc,
      studyId: activeStudy.id,
      patientId: activeStudy.patientId,
      modality: activeStudy.modality,
      aiFinding: activeStudy.primaryFinding,
      aiConfidence: activeStudy.primaryConfidence,
      decision: decisionType,
      radiologist: radiologistName,
      reasonCategory,
      notes:
        radiologistNotes.trim() ||
        (decisionType === "APPROVED"
          ? `Verified AI finding (${activeStudy.primaryFinding}) and Grad-CAM spatial localization.`
          : `Rejected AI finding (${activeStudy.primaryFinding}) after visual inspection of scan and Grad-CAM overlay.`),
      sha256: `7c82d1a0...${hashSuffix}`
    };

    setAuditLogs((prev) => [newEntry, ...prev]);
    setStudies((prev) =>
      prev.map((st) => (st.id === activeStudy.id ? { ...st, status: decisionType } : st))
    );
    setRadiologistNotes("");
    showToast(
      `Study ${activeStudy.id} marked as ${decisionType} by ${radiologistName} and recorded in the immutable Audit Log.`,
      decisionType === "APPROVED" ? "success" : "warning"
    );
  };

  // Simulate Orthanc DICOM Poller Service (Step 5)
  const handlePollOrthancServer = () => {
    if (isPollingOrthanc) return;
    setIsPollingOrthanc(true);
    const steps = [
      "Polling Orthanc REST API GET http://orthanc-pacs:8042/changes...",
      "Downloading new DICOM Series & stripping PHI tags (pydicom Safe Harbor)...",
      "Running MONAI Spacingd (1.0×1.0×1.5mm) & Z-Score Intensity Normalization...",
      "Executing PyTorch DenseNet-121 & MONAI 3D U-Net Sliding-Window Inference...",
      "Computing Grad-CAM spatial attribution hooks (features.norm5)..."
    ];
    let stepIndex = 0;
    setPollerStepText(steps[0]);

    const interval = setInterval(() => {
      stepIndex += 1;
      if (stepIndex < steps.length) {
        setPollerStepText(steps[stepIndex]);
      } else {
        clearInterval(interval);
        const newId = `ORTHANC-CXR-${1043 + studies.length}`;
        const pulledStudy = {
          ...INITIAL_STUDIES[0],
          id: newId,
          orthancUuid: `d120f884-92c1-440a-b812-${Date.now().toString().slice(-12)}`,
          patientId: `ANON-CXR-${9100 + studies.length}`,
          patientAgeSex: "61Y / F",
          studyDate: new Date().toISOString().replace("T", " ").slice(0, 16) + " UTC",
          primaryFinding: "Right Lower Zone Airspace Consolidation",
          primaryConfidence: 0.896,
          status: "PENDING"
        };
        setStudies((prev) => [pulledStudy, ...prev]);
        setSelectedStudyId(newId);
        setIsPollingOrthanc(false);
        setPollerStepText("");
        showToast(`Orthanc Poller ingested new study ${newId} and completed AI + Grad-CAM inference.`, "success");
      }
    }, 550);
  };

  // Handle Custom User Scan Upload
  const handleCustomScanUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result;
      if (typeof dataUrl !== "string") return;
      const newId = `UPLOAD-SCAN-${100 + studies.length}`;
      const customStudy = {
        ...INITIAL_STUDIES[0],
        id: newId,
        orthancUuid: `user-upload-${Date.now()}`,
        patientId: `ANON-UPLOAD-${studies.length + 1}`,
        patientAgeSex: "Custom Upload",
        studyDate: new Date().toISOString().replace("T", " ").slice(0, 16) + " UTC",
        modality: "DX",
        modalityName: `Uploaded Scan (${file.name.slice(0, 22)})`,
        datasetSource: "User Uploaded DICOM / Medical Radiograph",
        imageSrc: dataUrl,
        is3D: false,
        totalSlices: 1,
        defaultSlice: 1,
        roiCenter: { x: 0.34, y: 0.58 },
        roiRadii: { rx: 0.13, ry: 0.13 },
        primaryFinding: "Focal Parenchymal Opacity / ROI Detected",
        primaryConfidence: 0.884,
        status: "PENDING"
      };
      setStudies((prev) => [customStudy, ...prev]);
      setSelectedStudyId(newId);
      showToast(`Uploaded "${file.name}" — Grad-CAM and MONAI segmentation overlays generated.`, "success");
    };
    reader.readAsDataURL(file);
  };

  // Export Audit Logs as CSV
  const handleExportAuditCsv = () => {
    const headers = [
      "AuditID",
      "TimestampUTC",
      "StudyID",
      "PatientID",
      "Modality",
      "AIFinding",
      "AIConfidence",
      "Decision",
      "Radiologist",
      "ReasonCategory",
      "Notes"
    ];
    const rows = auditLogs.map((l) => [
      l.auditId,
      l.timestamp,
      l.studyId,
      l.patientId,
      l.modality,
      `"${l.aiFinding.replace(/"/g, '""')}"`,
      l.aiConfidence,
      l.decision,
      `"${l.radiologist.replace(/"/g, '""')}"`,
      `"${(l.reasonCategory || "").replace(/"/g, '""')}"`,
      `"${(l.notes || "").replace(/"/g, '""')}"`
    ]);
    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aiml02_radiologist_audit_log_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Exported Radiologist Audit Log as CSV.", "success");
  };

  const handleCopyAuditJson = () => {
    const text = JSON.stringify(auditLogs, null, 2);
    try {
      navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopiedAuditJson(true);
    setTimeout(() => setCopiedAuditJson(false), 2000);
  };

  const filteredAuditLogs = useMemo(() => {
    return auditLogs.filter((log) => {
      const matchesDecision = auditFilterDecision === "ALL" || log.decision === auditFilterDecision;
      const q = auditSearch.toLowerCase();
      const matchesSearch =
        !q ||
        log.studyId.toLowerCase().includes(q) ||
        log.patientId.toLowerCase().includes(q) ||
        log.aiFinding.toLowerCase().includes(q) ||
        log.radiologist.toLowerCase().includes(q) ||
        (log.notes || "").toLowerCase().includes(q);
      return matchesDecision && matchesSearch;
    });
  }, [auditLogs, auditFilterDecision, auditSearch]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col">
      {/* Mandatory Step 7 Regulatory & Educational Safety Banner */}
      <div className="bg-amber-950/90 border-b border-amber-700/60 px-4 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2 text-amber-200 font-medium">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong>FOR EDUCATION ONLY — NOT FOR CLINICAL USE:</strong> AI predictions (DenseNet-121 / MONAI 3D U-Net) and Grad-CAM heatmaps require licensed radiologist verification and must never be used for direct patient diagnosis.
          </span>
        </div>
        <button
          onClick={() => setActiveTab("safety")}
          className="text-amber-300 hover:text-amber-100 underline font-semibold shrink-0 flex items-center gap-1"
        >
          View Model Limits & Bias Documentation <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Top Navigation Header */}
      <header className="bg-slate-900/90 border-b border-slate-800 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-500/15 border border-teal-500/40 flex items-center justify-center text-teal-400">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 border border-teal-500/30">
                AIML-02
              </span>
              <h1 className="text-base sm:text-lg font-bold text-white tracking-tight">
                AI-Assisted Medical Imaging & Diagnosis Workstation
              </h1>
            </div>
            <p className="text-xs text-slate-400">
              Orthanc DICOM Server • PyTorch DenseNet-121 • MONAI 3D U-Net • Grad-CAM • DVC • Docker
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
          {[
            { id: "workstation", label: "PACS & AI Viewer", icon: Eye },
            { id: "orthanc_dvc", label: "Orthanc DICOM & DVC", icon: Database },
            { id: "audit_log", label: `Audit Log (${auditLogs.length})`, icon: ClipboardCheck },
            { id: "benchmarks", label: "Architecture & Metrics", icon: BarChart3 },
            { id: "safety", label: "Model Limits & 10-Min Deck", icon: BookOpen }
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  active
                    ? "bg-teal-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-4 right-4 z-50 max-w-md bg-slate-900 border border-teal-500/50 text-slate-100 px-4 py-3 rounded-xl shadow-2xl flex items-center gap-3 text-xs">
          <Sparkles className="w-4 h-4 text-teal-400 shrink-0" />
          <span>{toastMessage.msg}</span>
        </div>
      )}

      {/* Orthanc Live Poller Progress Bar */}
      {isPollingOrthanc && (
        <div className="bg-teal-950/80 border-b border-teal-700/50 px-4 py-2 flex items-center gap-3 text-xs text-teal-200">
          <RefreshCw className="w-4 h-4 text-teal-400 animate-spin shrink-0" />
          <span className="font-mono">{pollerStepText}</span>
        </div>
      )}

      {/* TAB 1: RADIOLOGIST PACS & AI DIAGNOSTIC WORKSTATION */}
      {activeTab === "workstation" && (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4">
          {/* Left Column: Orthanc Study Queue & Upload (3 cols) */}
          <div className="lg:col-span-3 flex flex-col gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-teal-400" />
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Orthanc PACS Worklist
                  </h2>
                </div>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-medium">
                  Port 8042 Online
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={handlePollOrthancServer}
                  disabled={isPollingOrthanc}
                  className="px-2.5 py-2 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isPollingOrthanc ? "animate-spin" : ""}`} />
                  <span>Pull Orthanc</span>
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-2.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Upload className="w-3.5 h-3.5 text-teal-400" />
                  <span>Upload Scan</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,.dcm"
                  onChange={handleCustomScanUpload}
                  className="hidden"
                />
              </div>

              {/* Study List */}
              <div className="flex flex-col gap-2 max-h-[380px] overflow-y-auto pr-1">
                {studies.map((st) => {
                  const isSelected = st.id === activeStudy.id;
                  return (
                    <button
                      key={st.id}
                      onClick={() => setSelectedStudyId(st.id)}
                      className={`text-left p-2.5 rounded-xl border transition-all ${
                        isSelected
                          ? "bg-teal-950/50 border-teal-500/70 shadow-sm"
                          : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="text-xs font-bold text-white font-mono">{st.id}</span>
                        <span
                          className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                            st.status === "APPROVED"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                              : st.status === "REJECTED"
                              ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                              : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                          }`}
                        >
                          {st.status}
                        </span>
                      </div>
                      <div className="text-xs font-medium text-teal-300 truncate">{st.modalityName}</div>
                      <div className="text-[11px] text-slate-400 truncate mt-0.5">{st.primaryFinding}</div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1.5 pt-1.5 border-t border-slate-800/80">
                        <span>{st.patientId} ({st.patientAgeSex})</span>
                        <span className="font-mono text-teal-400 font-semibold">
                          AI: {(st.primaryConfidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Active Study DICOM Metadata Summary */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-teal-400" /> DICOM Header Tags
                </span>
                <span className="text-[10px] font-mono text-slate-400">{activeStudy.modality}</span>
              </div>
              <div className="space-y-1.5 max-h-48 overflow-y-auto text-[11px]">
                {activeStudy.dicomTags.map((t) => (
                  <div key={t.tag} className="flex items-start justify-between gap-2 border-b border-slate-800/60 pb-1">
                    <span className="text-slate-400 font-mono shrink-0">{t.tag} {t.name}</span>
                    <span className="text-slate-200 font-mono text-right truncate">{t.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Center Column: Interactive Medical Scan, Grad-CAM & MONAI 3D U-Net Viewport (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col gap-3">
              {/* Viewport Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    onClick={() => setShowGradCam((v) => !v)}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 border transition ${
                      showGradCam
                        ? "bg-rose-500/20 text-rose-300 border-rose-500/50"
                        : "bg-slate-950 text-slate-400 border-slate-800"
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>Grad-CAM {showGradCam ? "ON" : "OFF"}</span>
                  </button>

                  <button
                    onClick={() => setShowSegMask((v) => !v)}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 border transition ${
                      showSegMask
                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50"
                        : "bg-slate-950 text-slate-400 border-slate-800"
                    }`}
                  >
                    <Cpu className="w-3.5 h-3.5" />
                    <span>3D U-Net ROI {showSegMask ? "ON" : "OFF"}</span>
                  </button>

                  <button
                    onClick={() => setShowCrosshair((v) => !v)}
                    className={`p-1.5 rounded-lg text-xs border transition ${
                      showCrosshair
                        ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                        : "bg-slate-950 text-slate-400 border-slate-800"
                    }`}
                    title="Toggle PACS Crosshair"
                  >
                    <Crosshair className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setZoom((z) => Math.min(2.2, Number((z + 0.2).toFixed(1))))}
                    className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-800"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setZoom((z) => Math.max(0.8, Number((z - 0.2).toFixed(1))))}
                    className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-800"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setInvertGrayscale((v) => !v)}
                    className={`p-1.5 rounded-lg border ${
                      invertGrayscale
                        ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                        : "bg-slate-950 text-slate-300 border-slate-800"
                    }`}
                    title="Invert DICOM Grayscale LUT"
                  >
                    <Contrast className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setFlipH((v) => !v)}
                    className={`p-1.5 rounded-lg border ${
                      flipH
                        ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                        : "bg-slate-950 text-slate-300 border-slate-800"
                    }`}
                    title="Flip Horizontal"
                  >
                    <FlipHorizontal className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => {
                      setZoom(1.0);
                      setInvertGrayscale(false);
                      setFlipH(false);
                      setWindowWidth(activeStudy.defaultWW);
                      setWindowLevel(activeStudy.defaultWL);
                      setCurrentSlice(activeStudy.defaultSlice);
                    }}
                    className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-800"
                    title="Reset Viewport"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Main Diagnostic Canvas Viewport */}
              <div className="relative bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center aspect-square">
                <div
                  className="w-full h-full flex items-center justify-center overflow-hidden"
                  style={{ transform: `scale(${zoom})`, transition: "transform 120ms ease-out" }}
                >
                  <canvas
                    ref={canvasRef}
                    onMouseMove={handleCanvasMouseMove}
                    className="max-w-full max-h-full object-contain cursor-crosshair"
                  />
                </div>

                {/* Top-Left Clinical PACS Overlay */}
                <div className="absolute top-2.5 left-2.5 pointer-events-none text-[11px] font-mono text-teal-300/90 bg-slate-950/75 px-2.5 py-1.5 rounded-lg border border-slate-800/80 space-y-0.5">
                  <div className="font-bold text-white">{activeStudy.id}</div>
                  <div>PT: {activeStudy.patientId} ({activeStudy.patientAgeSex})</div>
                  <div>MOD: {activeStudy.modalityName}</div>
                  {activeStudy.is3D && (
                    <div className="text-amber-300 font-semibold">
                      3D Axial Slice: {currentSlice} / {activeStudy.totalSlices}
                    </div>
                  )}
                </div>

                {/* Top-Right Model & Windowing Overlay */}
                <div className="absolute top-2.5 right-2.5 pointer-events-none text-right text-[11px] font-mono text-slate-300 bg-slate-950/75 px-2.5 py-1.5 rounded-lg border border-slate-800/80 space-y-0.5">
                  <div className="text-emerald-400 font-semibold">WW: {windowWidth} | WL: {windowLevel}</div>
                  <div>Zoom: {(zoom * 100).toFixed(0)}%</div>
                  <div>Latency: {activeStudy.inferenceTimeMs} ms</div>
                </div>

                {/* Bottom-Left Live Probe Readout */}
                <div className="absolute bottom-2.5 left-2.5 pointer-events-none text-[11px] font-mono text-slate-200 bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800">
                  XY: ({cursorProbe.x}, {cursorProbe.y}) • Intensity/HU: {cursorProbe.hu} • Grad-CAM: {(cursorProbe.camActivation * 100).toFixed(0)}%
                </div>

                {/* Bottom-Right Educational Watermark */}
                <div className="absolute bottom-2.5 right-2.5 pointer-events-none text-[10px] font-mono uppercase tracking-wider bg-amber-950/85 text-amber-300 px-2 py-0.5 rounded border border-amber-700/50">
                  Educational Use Only
                </div>
              </div>

              {/* Sliders: 3D Slice Scrubber, Grad-CAM Opacity, Window Width & Window Level */}
              <div className="bg-slate-950 border border-slate-800/90 rounded-xl p-3 space-y-2.5 text-xs">
                {activeStudy.is3D && (
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-amber-300 font-semibold shrink-0 w-36">
                      3D Axial Slice ({currentSlice}/{activeStudy.totalSlices})
                    </span>
                    <input
                      type="range"
                      min={1}
                      max={activeStudy.totalSlices}
                      value={currentSlice}
                      onChange={(e) => setCurrentSlice(Number(e.target.value))}
                      className="w-full accent-amber-400"
                    />
                  </div>
                )}

                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-300 font-medium shrink-0 w-36">
                    Grad-CAM Alpha ({Math.round(gradCamOpacity * 100)}%)
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.02}
                    value={gradCamOpacity}
                    onChange={(e) => setGradCamOpacity(Number(e.target.value))}
                    className="w-full accent-rose-500"
                  />
                  <select
                    value={gradCamColormap}
                    onChange={(e) => setGradCamColormap(e.target.value)}
                    className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-200"
                  >
                    <option value="jet">Jet LUT</option>
                    <option value="inferno">Inferno</option>
                    <option value="viridis">Viridis</option>
                  </select>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 border-t border-slate-800/80">
                  <div>
                    <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                      <span>Window Width (WW)</span>
                      <span className="font-mono text-slate-200">{windowWidth}</span>
                    </div>
                    <input
                      type="range"
                      min={100}
                      max={3000}
                      step={25}
                      value={windowWidth}
                      onChange={(e) => setWindowWidth(Number(e.target.value))}
                      className="w-full accent-teal-400"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                      <span>Window Level / Center (WL)</span>
                      <span className="font-mono text-slate-200">{windowLevel}</span>
                    </div>
                    <input
                      type="range"
                      min={-1000}
                      max={1000}
                      step={20}
                      value={windowLevel}
                      onChange={(e) => setWindowLevel(Number(e.target.value))}
                      className="w-full accent-teal-400"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: AI Confidence Scores, Segmentation Metrics & Radiologist Approve/Reject Sign-Off (4 cols) */}
          <div className="lg:col-span-4 flex flex-col gap-3">
            {/* AI Classification & MONAI Segmentation Summary Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col gap-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-teal-400">
                    Step 3 & 4 • Deep Learning Inference
                  </span>
                  <h3 className="text-sm font-bold text-white mt-0.5">{activeStudy.primaryFinding}</h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">{activeStudy.modelBackbone}</p>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-lg font-extrabold font-mono text-teal-400">
                    {(activeStudy.primaryConfidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-400">Confidence</div>
                </div>
              </div>

              {/* Multi-Label Pathology Probability Bars */}
              <div className="space-y-2 pt-1">
                {activeStudy.predictions.map((pred) => {
                  const pct = (pred.confidence * 100).toFixed(1);
                  const isHigh = pred.confidence >= pred.threshold;
                  return (
                    <div key={pred.finding} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className={isHigh ? "font-semibold text-white" : "text-slate-400"}>
                          {pred.finding}
                        </span>
                        <span
                          className={`font-mono text-xs ${
                            isHigh ? "text-teal-300 font-bold" : "text-slate-400"
                          }`}
                        >
                          {pct}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            pred.confidence >= 0.75
                              ? "bg-gradient-to-r from-teal-500 to-emerald-400"
                              : pred.confidence >= 0.5
                              ? "bg-amber-400"
                              : "bg-slate-700"
                          }`}
                          style={{ width: `${Math.min(100, pred.confidence * 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* MONAI 3D U-Net Volumetric Readout */}
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-center">
                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2">
                  <div className="text-[10px] text-slate-400">3D U-Net Volume</div>
                  <div className="text-xs font-bold font-mono text-emerald-400 mt-0.5">
                    {activeStudy.lesionVolumeCm3} cm³
                  </div>
                </div>
                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2">
                  <div className="text-[10px] text-slate-400">Cross-Section</div>
                  <div className="text-xs font-bold font-mono text-slate-200 mt-0.5">
                    {activeStudy.lesionAreaMm2} mm²
                  </div>
                </div>
                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2">
                  <div className="text-[10px] text-slate-400">Val Dice (DSC)</div>
                  <div className="text-xs font-bold font-mono text-teal-300 mt-0.5">
                    {activeStudy.diceScoreVal}
                  </div>
                </div>
              </div>

              <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-2.5 text-xs text-slate-300 leading-relaxed">
                <strong className="text-teal-300">AI Quantitative Impression: </strong>
                {activeStudy.clinicalSummary}
              </div>
            </div>

            {/* Step 6: Radiologist Review & Approve/Reject Audit Sign-Off */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Stethoscope className="w-4 h-4 text-teal-400" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Radiologist Verification & Audit Sign-Off
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-slate-400">Step 6</span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Reviewing Radiologist</label>
                  <select
                    value={radiologistName}
                    onChange={(e) => setRadiologistName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                  >
                    <option value="Dr. A. Sharma, MD (Radiology)">Dr. A. Sharma, MD (Radiology)</option>
                    <option value="Dr. M. Nair, MD (Neuroradiology & Chest)">Dr. M. Nair, MD (Neuroradiology & Chest)</option>
                    <option value="Dr. R. Kapoor, DNB (Thoracic Imaging)">Dr. R. Kapoor, DNB (Thoracic Imaging)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Clinical Concordance / Override Category</label>
                  <select
                    value={reasonCategory}
                    onChange={(e) => setReasonCategory(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                  >
                    <option value="True Positive — Concordant Grad-CAM & Anatomy">True Positive — Concordant Grad-CAM & Anatomy</option>
                    <option value="Approved with Minor Boundary Adjustment">Approved with Minor Boundary Adjustment</option>
                    <option value="Shortcut Learning / Confounder Artifact">Reject: Shortcut Learning / Confounder Artifact</option>
                    <option value="False Positive — Normal Anatomical Variant">Reject: False Positive — Normal Anatomical Variant</option>
                    <option value="Suboptimal Image Quality / Motion Artifact">Reject: Suboptimal Image Quality / Motion Artifact</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Radiologist Diagnostic Notes (Logged to Audit Trail)</label>
                  <textarea
                    rows={2}
                    value={radiologistNotes}
                    onChange={(e) => setRadiologistNotes(e.target.value)}
                    placeholder="Enter clinical observations, Grad-CAM verification notes, or reason for rejection..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2.5 pt-1">
                  <button
                    onClick={() => handleRadiologistDecision("APPROVED")}
                    className="py-2.5 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm transition"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Approve AI Diagnosis</span>
                  </button>
                  <button
                    onClick={() => handleRadiologistDecision("REJECTED")}
                    className="py-2.5 px-3 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm transition"
                  >
                    <XCircle className="w-4 h-4" />
                    <span>Reject AI Diagnosis</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: ORTHANC DICOM SERVER & DVC PREPROCESSING PIPELINE */}
      {activeTab === "orthanc_dvc" && (
        <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-bold uppercase text-teal-400">Step 1 & Step 2</span>
                <h2 className="text-base font-bold text-white mt-0.5">
                  DICOM Standard, Preprocessing & DVC Versioning
                </h2>
              </div>
              <span className="text-xs font-mono bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800 text-teal-300">
                dvc.yaml + params.yaml
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
                <div className="font-bold text-slate-200 mb-2 flex items-center gap-1.5">
                  <Sliders className="w-4 h-4 text-teal-400" /> Interactive Preprocessing & Augmentation Controls
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {[
                    { key: "resize224", label: "Resize 224×224 (CXR) / 128×128×64 (3D)" },
                    { key: "zScoreNorm", label: "Hounsfield Clip [-1000, 400] & Z-Score" },
                    { key: "randRotate", label: "MONAI RandRotate90d / Affine (±15°)" },
                    { key: "randFlip", label: "MONAI RandFlipd (Spatial Axis 0/1, p=0.5)" },
                    { key: "gaussianNoise", label: "MONAI RandGaussianNoised (σ=0.02)" }
                  ].map((item) => (
                    <label
                      key={item.key}
                      className="flex items-center gap-2 p-2 rounded-lg bg-slate-900 border border-slate-800 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={prepConfig[item.key]}
                        onChange={(e) =>
                          setPrepConfig((prev) => ({ ...prev, [item.key]: e.target.checked }))
                        }
                        className="accent-teal-500"
                      />
                      <span className="text-slate-300">{item.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-[11px] text-slate-300 overflow-x-auto">
                <div className="text-teal-400 font-bold mb-1.5"># dvc.yaml — Reproducible Data & Model Pipeline</div>
                <pre className="leading-relaxed">{`stages:
  preprocess_dicom:
    cmd: python backend/preprocess.py --config params.yaml --output data/processed
    deps: [backend/preprocess.py, backend/dicom_utils.py]
    params: [preprocess]
    outs: [data/processed]
  train_classifier:
    cmd: python backend/models/classifier.py --mode train --config params.yaml
    outs: [backend/models/checkpoints/densenet121_chexpert.pt]
    metrics: [metrics/classification_metrics.json]
  train_segmenter:
    cmd: python backend/models/segmenter.py --mode train --config params.yaml
    outs: [backend/models/checkpoints/monai_3d_unet_msd.pt]
    metrics: [metrics/segmentation_metrics.json]`}</pre>
              </div>
            </div>
          </div>

          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-bold uppercase text-teal-400">Step 5 & Step 7</span>
                <h2 className="text-base font-bold text-white mt-0.5">
                  Orthanc DICOM Server Poller & Docker Compose Stack
                </h2>
              </div>
              <button
                onClick={handlePollOrthancServer}
                className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold flex items-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Trigger Poller Now
              </button>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-[11px] text-slate-300 overflow-x-auto">
              <div className="text-emerald-400 font-bold mb-1.5"># backend/orthanc_service.py — Automated Study Poller</div>
              <pre className="leading-relaxed">{`async def poll_orthanc_changes(orthanc_url="http://orthanc-pacs:8042"):
    # 1. Poll Orthanc /changes endpoint for ChangeType == "NewStudy"
    resp = await client.get(f"{orthanc_url}/changes", params={"since": last_seq})
    for change in resp.json()["Changes"]:
        if change["ChangeType"] == "NewStudy":
            study_id = change["ID"]
            # 2. Pull raw DICOM instance bytes & strip PHI tags
            # 3. Run DenseNet-121 / MONAI 3D U-Net + Grad-CAM hooks
            # 4. Store result in FastAPI DB with status=PENDING_RADIOLOGIST_REVIEW`}</pre>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-[11px] text-slate-300 overflow-x-auto">
              <div className="text-sky-400 font-bold mb-1.5"># docker-compose.yml — Multi-Service Container Orchestration</div>
              <pre className="leading-relaxed">{`services:
  orthanc-pacs:
    image: jodogne/orthanc-plugins:1.12.4
    ports: ["4242:4242", "8042:8042"]
  ai-inference-backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - ORTHANC_URL=http://orthanc-pacs:8042
      - CLINICAL_DISCLAIMER=FOR_EDUCATION_ONLY_NOT_CLINICAL_USE
  react-radiology-viewer:
    build: ./frontend
    ports: ["3000:80"]`}</pre>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: IMMUTABLE RADIOLOGIST AUDIT LOG */}
      {activeTab === "audit_log" && (
        <div className="flex-1 p-4 flex flex-col gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="text-xs font-bold uppercase text-teal-400">Step 6 • Human-in-the-Loop Governance</span>
                <h2 className="text-base font-bold text-white mt-0.5">
                  Immutable Radiologist Decision Audit Log ({auditLogs.length} Records)
                </h2>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    value={auditSearch}
                    onChange={(e) => setAuditSearch(e.target.value)}
                    placeholder="Search study, patient, doctor..."
                    className="bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200"
                  />
                </div>

                <select
                  value={auditFilterDecision}
                  onChange={(e) => setAuditFilterDecision(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                >
                  <option value="ALL">All Decisions</option>
                  <option value="APPROVED">Approved Only</option>
                  <option value="REJECTED">Rejected Only</option>
                </select>

                <button
                  onClick={handleExportAuditCsv}
                  className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5" /> Export CSV
                </button>

                <button
                  onClick={handleCopyAuditJson}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold flex items-center gap-1.5"
                >
                  {copiedAuditJson ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedAuditJson ? "Copied JSON" : "Copy JSON"}</span>
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                    <th className="p-2.5 font-semibold">Audit ID & Timestamp</th>
                    <th className="p-2.5 font-semibold">Study & Patient</th>
                    <th className="p-2.5 font-semibold">AI Prediction & Score</th>
                    <th className="p-2.5 font-semibold">Decision</th>
                    <th className="p-2.5 font-semibold">Radiologist & Clinical Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {filteredAuditLogs.map((log) => (
                    <tr key={log.auditId} className="hover:bg-slate-950/40">
                      <td className="p-2.5 font-mono">
                        <div className="font-bold text-slate-200">{log.auditId}</div>
                        <div className="text-[11px] text-slate-400">{log.timestamp}</div>
                        <div className="text-[10px] text-slate-500">SHA256: {log.sha256}</div>
                      </td>
                      <td className="p-2.5 font-mono">
                        <div className="font-bold text-teal-300">{log.studyId}</div>
                        <div className="text-[11px] text-slate-400">
                          {log.patientId} • {log.modality}
                        </div>
                      </td>
                      <td className="p-2.5">
                        <div className="font-semibold text-white">{log.aiFinding}</div>
                        <div className="text-[11px] font-mono text-teal-400">
                          Confidence: {(log.aiConfidence * 100).toFixed(1)}%
                        </div>
                      </td>
                      <td className="p-2.5">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold ${
                            log.decision === "APPROVED"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                              : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          }`}
                        >
                          {log.decision === "APPROVED" ? (
                            <CheckCircle2 className="w-3 h-3" />
                          ) : (
                            <XCircle className="w-3 h-3" />
                          )}
                          {log.decision}
                        </span>
                      </td>
                      <td className="p-2.5 max-w-md">
                        <div className="font-semibold text-slate-200">{log.radiologist}</div>
                        <div className="text-[11px] text-teal-300/90">{log.reasonCategory}</div>
                        <div className="text-[11px] text-slate-400 mt-0.5">{log.notes}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: ARCHITECTURE & BENCHMARK PERFORMANCE METRICS */}
      {activeTab === "benchmarks" && (
        <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
            <div>
              <span className="text-xs font-bold uppercase text-teal-400">Step 3A • Classification Benchmark</span>
              <h3 className="text-sm font-bold text-white mt-0.5">
                DenseNet-121 vs. EfficientNet-B4 Per-Class AUROC (NIH ChestX-ray14 / CheXpert)
              </h3>
            </div>
            <div className="relative h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={BENCHMARK_AUROC_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="pathology" stroke="#94a3b8" fontSize={11} />
                  <YAxis domain={[0.75, 1.0]} stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Bar dataKey="DenseNet121" name="DenseNet-121 (AUROC)" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="EfficientNetB4" name="EfficientNet-B4 (AUROC)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
            <div>
              <span className="text-xs font-bold uppercase text-teal-400">Step 3B • Volumetric Segmentation</span>
              <h3 className="text-sm font-bold text-white mt-0.5">
                MONAI 3D Residual U-Net Validation Dice Progression (BraTS & MSD Lung)
              </h3>
            </div>
            <div className="relative h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={UNET_TRAINING_HISTORY}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={11} />
                  <YAxis domain={[0.1, 1.0]} stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Line type="monotone" dataKey="BrainDice" name="Brain MRI Whole Tumor Dice" stroke="#10b981" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="LungDice" name="Lung CT Nodule Dice" stroke="#38bdf8" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="ValLoss" name="DiceCE Validation Loss" stroke="#f43f5e" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: MODEL LIMITATIONS, BIAS ANALYSIS & 10-MINUTE PRESENTATION DECK */}
      {activeTab === "safety" && (
        <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center gap-2 text-amber-400">
              <AlertTriangle className="w-5 h-5" />
              <h2 className="text-base font-bold text-white">
                Step 7: Model Limitations, Bias Analysis & Clinical Safety Notice
              </h2>
            </div>
            <div className="p-3 rounded-xl bg-amber-950/60 border border-amber-700/60 text-xs text-amber-200 leading-relaxed">
              <strong>MANDATORY NOTICE:</strong> This system is designed <strong>FOR EDUCATION ONLY, NOT FOR CLINICAL USE</strong>. All outputs must be reviewed by a qualified radiologist.
            </div>
            <div className="space-y-2.5 text-xs text-slate-300 leading-relaxed">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <strong className="text-teal-300">1. Shortcut Learning & Hardware Confounders:</strong> Deep CNNs trained on public chest X-ray datasets (NIH ChestX-ray14, CheXpert) can latch onto hospital tokens, portable AP projection markers, chest tubes, or ICU central lines as spurious predictors of pathology. Grad-CAM overlays allow the reviewing radiologist to spot and reject confounder-driven predictions immediately.
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <strong className="text-teal-300">2. Demographic & Scanner Domain Shift:</strong> Models trained on adult tertiary-care academic cohorts show reduced sensitivity on pediatric scans and across unseen CT reconstruction kernels or MRI field strengths (1.5T vs. 3.0T).
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <strong className="text-teal-300">3. Automated NLP Ground-Truth Label Noise:</strong> CheXpert and NIH ChestX-ray14 labels are extracted from free-text radiology reports via rule-based NLP, carrying ~5–8% uncertainty on overlapping findings (e.g., Consolidation vs. Infiltration).
              </div>
            </div>
          </div>

          <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center gap-2 text-teal-400">
              <Terminal className="w-5 h-5" />
              <h2 className="text-base font-bold text-white">
                Deliverables Checklist: 10-Minute Presentation & Intern Reflection
              </h2>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              {[
                {
                  time: "0:00 – 1:30",
                  title: "1. Problem & Clinical Motivation",
                  desc: "Rising radiology scan backlog and why black-box AI is unsafe without spatial Grad-CAM attribution and PACS workflow integration."
                },
                {
                  time: "1:30 – 3:30",
                  title: "2. DICOM Standard, MONAI Preprocessing & DVC",
                  desc: "Parsing DICOM headers, HIPAA Safe Harbor de-identification, Hounsfield Unit windowing, 3D isotropic spacing resampling, and DVC pipeline tracking."
                },
                {
                  time: "3:30 – 5:30",
                  title: "3. Transfer Learning Classifier + MONAI 3D U-Net + Grad-CAM",
                  desc: "DenseNet-121 / EfficientNet-B4 multi-label classification (Macro AUROC 0.864) and 5-stage Residual MONAI 3D U-Net (BraTS Dice 0.894, Lung Dice 0.838)."
                },
                {
                  time: "5:30 – 8:00",
                  title: "4. Live Demo (Orthanc PACS -> AI -> React Workstation)",
                  desc: "Ingesting studies from Orthanc (Port 8042), scrubbing 3D slices, adjusting WW/WL, inspecting Grad-CAM heatmaps, and recording Approve/Reject audit logs."
                },
                {
                  time: "8:00 – 10:00",
                  title: "5. Bias Analysis, Challenges & Next Steps",
                  desc: "Mitigating ICU shortcut learning via human-in-the-loop rejection logs, temperature calibration (ECE 0.031), and roadmap to DICOM SR export."
                }
              ].map((slide) => (
                <div key={slide.title} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-start gap-3">
                  <span className="font-mono text-[11px] text-teal-400 font-bold shrink-0 bg-teal-950/60 px-2 py-0.5 rounded border border-teal-800">
                    {slide.time}
                  </span>
                  <div>
                    <div className="font-bold text-white">{slide.title}</div>
                    <div className="text-slate-400 mt-0.5">{slide.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
