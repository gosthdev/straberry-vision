import sys
from pathlib import Path
import shutil
import os
from typing import List
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn
import cv2
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.core import SGSNet, Config, non_max_suppression

app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://strawberry-vision.netlify.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https://.*\.ngrok-free\.app",  # Permite cualquier subdominio de ngrok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("app/static/uploads")
RESULTS_DIR = Path("app/static/results")
STATIC_DIR = Path("app/static")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files (solo para servir imágenes procesadas)
app.mount("/uploads", StaticFiles(directory="app/static/uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="app/static/results"), name="results")

# Model Service
class ModelService:
    def __init__(self):
        self.model = None
        self.device = Config.DEVICE
        self.load_model()

    def load_model(self):
        checkpoint_path = Path("src/data/processed/models/best_model.pth")
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            # Try to find any .pth file in the directory
            models_dir = Path("src/data/processed/models")
            if models_dir.exists():
                pths = list(models_dir.glob("*.pth"))
                if pths:
                    checkpoint_path = pths[0]
                    print(f"Using alternative checkpoint: {checkpoint_path}")
                else:
                    print("No checkpoints found.")
                    return
            else:
                print("Models directory not found.")
                return

        try:
            self.model = SGSNet(Config.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, image_path: Path):
        if self.model is None:
            # Return empty detections if model is not loaded
            image_bgr = cv2.imread(str(image_path))
            return [], image_bgr

        # Preprocess
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        original_h, original_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(image_rgb, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        resized_rgb = resized_rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalised = (resized_rgb - mean) / std

        tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model(tensor)

        # NMS
        batch_detections = non_max_suppression(
            predictions,
            conf_threshold=0.4, # Use config or default
            iou_threshold=Config.IOU_THRESHOLD,
        )[0]

        boxes = batch_detections[0].cpu().numpy()
        labels = batch_detections[1].cpu().numpy()
        scores = batch_detections[2].cpu().numpy()

        # Convert to pixels
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * original_w)
            y1 = int((cy - h / 2) * original_h)
            x2 = int((cx + w / 2) * original_w)
            y2 = int((cy + h / 2) * original_h)

            x1 = max(0, min(original_w - 1, x1))
            y1 = max(0, min(original_h - 1, y1))
            x2 = max(0, min(original_w - 1, x2))
            y2 = max(0, min(original_h - 1, y2))

            class_idx = int(label)
            class_name = Config.CLASS_NAMES[class_idx] if 0 <= class_idx < len(Config.CLASS_NAMES) else f"cls_{class_idx}"
            detections.append({
                "class": class_name,
                "score": float(score),
                "box": [x1, y1, x2, y2]
            })

        return detections, image_bgr

    def annotate(self, image_bgr, detections):
        annotated = image_bgr.copy()
        colour_palette = [
            (0, 0, 255),      # flowering -> red
            (0, 255, 0),      # growing_g -> green
            (255, 255, 0),    # growing_w -> cyan
            (0, 255, 255),    # nearly_m -> yellow
            (255, 0, 255),    # mature -> magenta
        ]

        for det in detections:
            class_name = det["class"]
            score = det["score"]
            x1, y1, x2, y2 = det["box"]
            
            colour = colour_palette[Config.CLASS_NAMES.index(class_name) % len(colour_palette)] if class_name in Config.CLASS_NAMES else (200, 200, 200)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            label = f"{class_name}: {score:.2f}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - h - baseline - 2)), (x1 + w, y1), colour, thickness=-1)
            cv2.putText(annotated, label, (x1, max(10, y1 - baseline - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated

model_service = ModelService()


@app.get("/")
async def health_check():
    """Health endpoint for backend-only deployments."""
    return {"status": "ok", "message": "Strawberry Vision API is running"}


# Serve a small dynamic config.js so clients requesting /static/js/config.js get a valid file
# API URL is taken from env var `API_PUBLIC_URL` (set this to your ngrok or public URL)
@app.get("/static/js/config.js")
async def serve_config_js():
    api_url = os.environ.get("API_PUBLIC_URL", "")
    js = f"window.APP_CONFIG = {{\n    API_URL: \"{api_url}\"\n}};\n"
    return Response(content=js, media_type="application/javascript")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Skipping invalid image: {file.filename}")
            continue

        # Save as WebP
        unique_filename = f"{uuid.uuid4()}.webp"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save with 90% quality
        cv2.imwrite(str(file_path), img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
            
        uploaded_files.append({
            "original_filename": file.filename,
            "server_filename": unique_filename,
            "url": f"/uploads/{unique_filename}"
        })
    return {"files": uploaded_files}

@app.post("/analyze")
async def analyze_images(filenames: List[str] = Body(...), original_filenames: List[str] = Body(...)):
    results = []
    for i, filename in enumerate(filenames):
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            continue
            
        detections, image_bgr = model_service.predict(file_path)
        annotated_img = model_service.annotate(image_bgr, detections)
        
        annotated_filename = f"annotated_{filename}"
        annotated_path = RESULTS_DIR / annotated_filename
        cv2.imwrite(str(annotated_path), annotated_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
        
        # Calculate average confidence
        avg_conf = 0
        if detections:
            avg_conf = sum(d["score"] for d in detections) / len(detections)
            
        results.append({
            "filename": original_filenames[i] if i < len(original_filenames) else filename,
            "detections": detections,
            "count": len(detections),
            "avg_confidence": avg_conf * 100, # Convert to percentage
            "thumbnail_url": f"/results/{annotated_filename}",
            "annotated_url": f"/results/{annotated_filename}"
        })
    return {"results": results}

@app.post("/cleanup")
async def cleanup_files(filenames: List[str] = Body(...)):
    for filename in filenames:
        # Sanitize filename just in case
        filename = Path(filename).name
        
        # Delete uploaded file
        upload_path = UPLOAD_DIR / filename
        if upload_path.exists():
            try:
                os.remove(upload_path)
            except Exception as e:
                print(f"Error deleting {upload_path}: {e}")
        
        # Delete result file
        annotated_filename = f"annotated_{filename}"
        result_path = RESULTS_DIR / annotated_filename
        if result_path.exists():
            try:
                os.remove(result_path)
            except Exception as e:
                print(f"Error deleting {result_path}: {e}")
                
    return {"message": "Cleanup successful"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
