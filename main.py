import io
import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(
    title="PhytoScan AI Engine v2.0",
    description="Agentic multi-stage plant pathology pipeline with blur detection, ONNX inference, and agronomic prescriptions.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. KNOWLEDGE BASE: Agronomic Action Database
# ---------------------------------------------------------
TREATMENT_REGISTRY: Dict[str, Dict] = {
    "early_blight": {
        "pathogen_type": "Fungal (Alternaria solani)",
        "urgency": "High",
        "organic_treatment": "Apply copper-based fungicides or Bacillus subtilis sprays. Remove infected lower leaves.",
        "chemical_treatment": "Mancozeb (75% WP) or Chlorothalonil applied at 7-10 day intervals.",
        "cultural_practice": "Ensure drip irrigation to prevent leaf wetness; maintain crop rotation away from Solanaceae."
    },
    "late_blight": {
        "pathogen_type": "Oomycete (Phytophthora infestans)",
        "urgency": "Critical",
        "organic_treatment": "Immediate destruction of infected plant debris; Bordeaux mixture sprays.",
        "chemical_treatment": "Metalaxyl-M + Mancozeb or Cymoxanil spray.",
        "cultural_practice": "Improve field drainage and plant certified disease-free seeds."
    },
    "powdery_mildew": {
        "pathogen_type": "Fungal (Erysiphales)",
        "urgency": "Medium",
        "organic_treatment": "Potassium bicarbonate spray, neem oil (0.5%), or diluted milk spray (1:9 ratio).",
        "chemical_treatment": "Sulfur-based dusts or Myclobutanil.",
        "cultural_practice": "Prune foliage to maximize air circulation and direct sunlight penetration."
    },
    "bacterial_spot": {
        "pathogen_type": "Bacterial (Xanthomonas spp.)",
        "urgency": "High",
        "organic_treatment": "Fixed copper sprays mixed with bio-fungicides.",
        "chemical_treatment": "Copper sulfate + Streptomycin/Oxytetracycline formulations where permitted.",
        "cultural_practice": "Avoid working in fields when foliage is wet; sanitize pruning shears with 70% alcohol."
    },
    "abiotic_chlorosis": {
        "pathogen_type": "Non-infectious (Nutrient/Environmental Stress)",
        "urgency": "Low",
        "organic_treatment": "Foliar spray of balanced organic sea-kelp extract or compost tea.",
        "chemical_treatment": "Chelated iron (Fe-EDTA) or foliar urea (1-2%) based on soil pH test.",
        "cultural_practice": "Check soil compaction and regulate irrigation frequency to prevent root suffocation."
    },
    "healthy_senescent": {
        "pathogen_type": "Natural Senescence",
        "urgency": "None",
        "organic_treatment": "No treatment required. Prune old bottom leaves to redirect plant energy to new shoots.",
        "chemical_treatment": "None.",
        "cultural_practice": "Normal maintenance schedule."
    }
}

# ---------------------------------------------------------
# 2. ONNX INFERENCE ENGINE
# ---------------------------------------------------------
ONNX_MODEL_PATH = "models/plant_disease_yolov10.onnx"
CLASS_NAMES = [
    "healthy_green", "healthy_senescent", "abiotic_chlorosis",
    "early_blight", "late_blight", "powdery_mildew", "bacterial_spot", "rust"
]

try:
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    print(f"Loaded ONNX Model. Active Provider: {session.get_providers()[0]}")
except Exception as e:
    session = None
    print(f"ONNX Model not loaded from {ONNX_MODEL_PATH}. Running in mock mode. Error: {e}")

# ---------------------------------------------------------
# 3. PYDANTIC SCHEMAS
# ---------------------------------------------------------
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

class DetectedLesion(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox
    is_pathogen: bool

class TreatmentPlan(BaseModel):
    pathogen_type: str
    urgency: str
    organic_treatment: str
    chemical_treatment: str
    cultural_practice: str

class DiagnosisV2Response(BaseModel):
    status: str
    processing_time_ms: float
    quality_score: Dict[str, float]
    severity_index: str
    total_lesions: int
    detections: List[DetectedLesion]
    prescriptions: Dict[str, TreatmentPlan]
    advice: str

# ---------------------------------------------------------
# 4. PREPROCESSING & VALIDATION HELPERS
# ---------------------------------------------------------
def assess_image_quality(cv_img: np.ndarray) -> Dict[str, float]:
    """Calculates blur via Laplacian variance and mean brightness."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    return {
        "blur_variance": round(blur_score, 2),
        "brightness": round(brightness, 2)
    }

def preprocess_for_onnx(cv_img: np.ndarray, target_size: int = 640):
    """Resizes, transposes, and normalizes image for ONNX model input."""
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Letterbox resize maintaining aspect ratio
    scale = min(target_size / h, target_size / w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    
    tensor = canvas.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, scale, dx, dy

# ---------------------------------------------------------
# 5. DIAGNOSIS ENDPOINT
# ---------------------------------------------------------
@app.post("/api/v2/diagnose", response_model=DiagnosisV2Response)
async def diagnose_leaf_v2(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.35, ge=0.1, le=1.0)
):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Corrupted image buffer.")

    # Gate 1: Check image quality (blur & lighting)
    quality = assess_image_quality(cv_img)
    if quality["blur_variance"] < 40.0:
        raise HTTPException(
            status_code=422, 
            detail=f"Image is too blurry (Laplacian variance: {quality['blur_variance']} < 40.0). Retake with focus on the leaf."
        )

    detections: List[DetectedLesion] = []
    
    # Gate 2: ONNX Model Inference
    if session:
        tensor, scale, dx, dy = preprocess_for_onnx(cv_img, 640)
        outputs = session.run(None, {input_name: tensor})
        predictions = outputs[0]  # Shape: [1, num_boxes, 6] -> [x1, y1, x2, y2, conf, class_id]
        
        orig_h, orig_w = cv_img.shape[:2]

        for pred in predictions[0]:
            conf = float(pred[4])
            if conf >= conf_threshold:
                cls_id = int(pred[5])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                
                # Rescale boxes back to original image space
                x1 = max(0, (pred[0] - dx) / scale)
                y1 = max(0, (pred[1] - dy) / scale)
                x2 = min(orig_w, (pred[2] - dx) / scale)
                y2 = min(orig_h, (pred[3] - dy) / scale)
                
                is_pathogen = cls_name not in ["healthy_green", "healthy_senescent", "abiotic_chlorosis"]

                detections.append(DetectedLesion(
                    class_name=cls_name,
                    confidence=round(conf, 3),
                    box=BoundingBox(
                        x1=round(x1, 1), y1=round(y1, 1),
                        x2=round(x2, 1), y2=round(y2, 1),
                        width=round(x2 - x1, 1),
                        height=round(y2 - y1, 1)
                    ),
                    is_pathogen=is_pathogen
                ))

    # Gate 3: Severity Calculation & Remedy Association
    pathogenic_lesions = [d for d in detections if d.is_pathogen]
    num_lesions = len(pathogenic_lesions)
    
    if num_lesions == 0:
        severity = "None (Healthy / Benign)"
    elif num_lesions <= 2:
        severity = "Mild (Localized infection, early stage)"
    elif num_lesions <= 6:
        severity = "Moderate (Multi-lesion spread across foliage)"
    else:
        severity = "Severe (Advanced outbreak, immediate isolation required)"

    prescriptions: Dict[str, TreatmentPlan] = {}
    unique_detected_classes = {d.class_name for d in detections}
    
    for cls in unique_detected_classes:
        if cls in TREATMENT_REGISTRY:
            data = TREATMENT_REGISTRY[cls]
            prescriptions[cls] = TreatmentPlan(
                pathogen_type=data["pathogen_type"],
                urgency=data["urgency"],
                organic_treatment=data["organic_treatment"],
                chemical_treatment=data["chemical_treatment"],
                cultural_practice=data["cultural_practice"]
            )

    advice = "Continue standard irrigation and nutrient cycles."
    if pathogenic_lesions:
        names = ", ".join({d.class_name.replace('_', ' ').title() for d in pathogenic_lesions})
        advice = f"Action required: Detected active lesions of {names}. Refer to targeted treatment prescriptions."

    total_time = round((time.time() - start_time) * 1000, 2)

    return DiagnosisV2Response(
        status="success",
        processing_time_ms=total_time,
        quality_score=quality,
        severity_index=severity,
        total_lesions=len(detections),
        detections=detections,
        prescriptions=prescriptions,
        advice=advice
    )
