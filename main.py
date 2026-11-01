import io
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI(
    title="Plant Disease Object Detection API",
    description="Backend API detecting multiple leaf diseases, healthy senescent leaves, and abiotic stress.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once at startup (YOLOv8/v9/v10 or RT-DETR weights)
# In production, replace 'best.pt' with 'best.onnx' using ONNXRuntime for faster throughput
MODEL_PATH = "best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model not loaded from {MODEL_PATH}. Make sure weights exist. Error: {e}")

# Pydantic Response Schemas
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    is_pathogen: bool

class AnalysisResponse(BaseModel):
    success: bool
    inference_time_ms: float
    total_detections: int
    detections: List[Detection]
    diagnostic_summary: str

# Helper to identify whether a predicted class is an active disease or a benign state
PATHOGEN_CLASSES = {
    "early_blight", "late_blight", "powdery_mildew", 
    "bacterial_spot", "rust", "leaf_mold"
}

BENIGN_CLASSES = {
    "healthy_green", "healthy_senescent", "abiotic_chlorosis"
}

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=AnalysisResponse)
async def predict_plant_disease(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.40, ge=0.1, le=1.0)
):
    if not model:
        raise HTTPException(status_code=503, detail="Model weights are not loaded.")

    # 1. Validate file extension and format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    start_time = time.time()
    
    # 2. Read image bytes into PIL
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    # 3. Run Inference
    results = model.predict(
        source=image, 
        conf=confidence_threshold,
        imgsz=640,
        verbose=False
    )[0]

    inference_ms = round((time.time() - start_time) * 1000, 2)
    detections: List[Detection] = []
    
    # 4. Parse Bounding Boxes & Confidence Scores
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        class_name = results.names[cls_id]
        conf = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        is_pathogen = class_name.lower() in PATHOGEN_CLASSES

        detections.append(Detection(
            class_id=cls_id,
            class_name=class_name,
            confidence=round(conf, 4),
            box=BoundingBox(
                x1=round(coords[0], 2),
                y1=round(coords[1], 2),
                x2=round(coords[2], 2),
                y2=round(coords[3], 2)
            ),
            is_pathogen=is_pathogen
        ))

    # 5. Formulate Diagnostic Summary
    if len(detections) == 0:
        summary = "No leaf lesions or plant features detected. Ensure a clear plant image is uploaded."
    else:
        pathogens_found = list({d.class_name for d in detections if d.is_pathogen})
        benign_found = list({d.class_name for d in detections if not d.is_pathogen})
        
        if pathogens_found:
            summary = f"Detected {len(detections)} area(s) of interest. Active pathogens: {', '.join(pathogens_found)}."
            if "healthy_senescent" in benign_found or "abiotic_chlorosis" in benign_found:
                summary += " Also observed natural yellowing/abiotic stress patterns."
        else:
            summary = f"No active pathogens detected. Leaf exhibits: {', '.join(benign_found)}."

    return AnalysisResponse(
        success=True,
        inference_time_ms=inference_ms,
        total_detections=len(detections),
        detections=detections,
        diagnostic_summary=summary
    )
