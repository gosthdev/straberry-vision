import os
import base64
from typing import Dict, List

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inference_service import infer_from_bytes

# =========================
# RUTAS DE PROYECTO
# =========================
API_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(API_DIR, "templates")
STATIC_DIR = os.path.join(API_DIR, "static")

app = FastAPI(title="Strawberry Vision API")

# Archivos estáticos (CSS, JS, imágenes)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Sistema de templates (Jinja2)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# =========================
# ENDPOINTS HTML (PÁGINAS)
# =========================

@app.get("/", response_class=HTMLResponse, name="home")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/single", response_class=HTMLResponse, name="single_image_page")
def single_image_page(request: Request):
    return templates.TemplateResponse(
        "single_image.html",
        {"request": request},
    )


@app.get("/batch", response_class=HTMLResponse, name="batch_page")
def batch_page(request: Request):
    return templates.TemplateResponse(
        "batch.html",
        {"request": request},
    )


@app.get("/results", response_class=HTMLResponse, name="results_page")
def results_page(request: Request):
    return templates.TemplateResponse(
        "results.html",
        {"request": request},
    )


@app.get("/about", response_class=HTMLResponse, name="about_page")
def about_page(request: Request):
    return templates.TemplateResponse(
        "about.html",
        {"request": request},
    )

@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "API de Strawberry Vision funcionando"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
) -> JSONResponse:
    try:
        contents = await file.read()
        result = infer_from_bytes(contents, conf_threshold=0.4)

        annotated_bgr = result.pop("annotated_image_bgr")
        success, buffer = cv2.imencode(".png", annotated_bgr)
        if not success:
            raise RuntimeError("No se pudo codificar la imagen anotada.")

        image_base64 = base64.b64encode(buffer).decode("utf-8")
        result["annotated_image_base64"] = image_base64

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar la imagen: {e}",
        ) from e

@app.post("/predict-batch")
async def predict_batch(
    lote_id: str = Form(...),
    location: str = Form(""),
    description: str = Form(""),
    files: List[UploadFile] = File(...),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron imágenes.")

    batch_results = []

    for f in files:
        try:
            contents = await f.read()
            result = infer_from_bytes(contents, conf_threshold=0.4)

            annotated_bgr = result.pop("annotated_image_bgr")
            success, buffer = cv2.imencode(".png", annotated_bgr)
            if not success:
                raise RuntimeError("No se pudo codificar la imagen anotada.")

            image_base64 = base64.b64encode(buffer).decode("utf-8")

            batch_results.append(
                {
                    "filename": f.filename,
                    **result,
                    "annotated_image_base64": image_base64,
                }
            )
        except Exception as e:
            batch_results.append(
                {
                    "filename": f.filename,
                    "error": str(e),
                }
            )

    response = {
        "batch_meta": {
            "lote_id": lote_id,
            "location": location,
            "description": description,
            "conf_threshold": 0.4,
        },
        "num_files": len(batch_results),
        "results": batch_results,
    }

    return JSONResponse(content=response)