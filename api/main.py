import os
import base64
import uuid
from pathlib import Path
from typing import Dict, List
import shutil
import subprocess
import sys
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference_service import infer_from_bytes

# =========================
# RUTAS DE PROYECTO
# =========================
API_DIR = Path(__file__).parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"
PROJECT_ROOT = API_DIR.parent 

# Directorios de Spark
BATCH_INCOMING = Path(PROJECT_ROOT) / "test" / "data" / "batch_incoming"
BATCH_OUTPUT = Path(PROJECT_ROOT) / "test" / "data" / "batch_outputs"
BATCH_INCOMING.mkdir(parents=True, exist_ok=True)
BATCH_OUTPUT.mkdir(parents=True, exist_ok=True)

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

# =========================
# ENDPOINT BATCH (NUEVO - ENCOLA PARA SPARK)
# =========================

@app.post("/predict-batch")
async def predict_batch(
    lote_id: str = Form(...),
    location: str = Form(""),
    description: str = Form(""),
    files: List[UploadFile] = File(...),
) -> JSONResponse:
    """Procesa lote de imágenes con PySpark"""
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron imágenes.")

    batch_id = str(uuid.uuid4())[:8]
    batch_dir = BATCH_INCOMING / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []

    # 1. GUARDAR IMÁGENES
    print(f"\n[1/3] Guardando {len(files)} imágenes para lote {batch_id}...")
    for f in files:
        try:
            contents = await f.read()
            
            if not f.content_type.startswith("image/"):
                errors.append({"filename": f.filename, "error": "No es imagen válida"})
                continue

            # Convertir a WEBP
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                errors.append({"filename": f.filename, "error": "Imagen corrupta"})
                continue

            filename = Path(f.filename).stem + ".webp"
            filepath = batch_dir / filename

            success, buffer = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 85])
            if not success:
                errors.append({"filename": f.filename, "error": "Error encoding WEBP"})
                continue

            with open(filepath, "wb") as fp:
                fp.write(buffer.tobytes())

            saved_files.append(f.filename)
            print(f"  ✓ {f.filename}")

        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    if not saved_files:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Error al guardar imágenes: {errors}")

    # 2. EJECUTAR PYSPARK SINCRONICAMENTE
    print(f"\n[2/3] Ejecutando PySpark para {len(saved_files)} imágenes...")
    
    env = os.environ.copy()
    env["BATCH_INPUT_DIR"] = str(BATCH_INCOMING)
    env["BATCH_OUTPUT_DIR"] = str(BATCH_OUTPUT)
    env["BATCH_MODEL_PATH"] = "src/data/processed/models/best_model.pth"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    spark_submit = "spark-submit"
    app_path = PROJECT_ROOT / "src" / "core" / "batch.py"  # ← AHORA SÍ FUNCIONA
    
    command = [
        spark_submit,
        "--master", "local[*]",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={sys.executable}",
        "--conf", f"spark.executorEnv.PYSPARK_DRIVER_PYTHON={sys.executable}",
        "--conf", "spark.driver.memory=2g",
        "--conf", "spark.executor.memory=1g",
        "--conf", "spark.sql.execution.arrow.pyspark.enabled=true",
        str(app_path),
    ]

    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0:
            print(f"ERROR en PySpark:\n{result.stderr}")    
            raise HTTPException(
                status_code=500, 
                detail=f"Error en PySpark: {result.stderr[:500]}"
            )
        
        print(f"  ✓ PySpark completado exitosamente")
        print(f"Stdout:\n{result.stdout[:500]}")

    except subprocess.TimeoutExpired:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise HTTPException(
            status_code=504, 
            detail="PySpark tardó demasiado (>5min). Intenta con menos imágenes."
        )
    except Exception as e:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error ejecutando PySpark: {str(e)}")

    # 3. CARGAR RESULTADOS
    print(f"\n[3/3] Cargando resultados...")
    
    batch_output_dir = BATCH_OUTPUT / batch_id / "parquet_data"
    if not batch_output_dir.exists():
        raise HTTPException(
            status_code=500, 
            detail="PySpark no generó resultados"
        )

    parquet_files = list(batch_output_dir.glob("*.parquet"))
    if not parquet_files:
        raise HTTPException(
            status_code=500, 
            detail="No se encontraron archivos parquet"
        )

    # Convertir parquet a resultados legibles
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_files[0])
        
        results = []
        for idx, row in df.iterrows():
            results.append({
                "filename": Path(row['path']).name,
                "num_detections": len(row['detections']) if row['detections'] else 0,
                "detections": row['detections'] if row['detections'] else [],
            })
        
        print(f"  ✓ {len(results)} resultados cargados")

    except Exception as e:
        print(f"Advertencia: No se pudo parsear parquet: {e}")
        results = [{"filename": f, "num_detections": 0} for f in saved_files]

    return JSONResponse(
        content={
            "batch_id": batch_id,
            "status": "completed",
            "message": f"Lote procesado exitosamente: {len(saved_files)} imágenes",
            "num_files": len(saved_files),
            "results": results,
            "download_url": f"/download-batch/{batch_id}",
            "metadata": {
                "lote_id": lote_id,
                "location": location,
                "description": description,
            }
        },
        status_code=200
    )

# =========================
# ENDPOINT PARA MONITOREAR PROGRESO Y DESCARGAR RESULTADOS
# =========================

@app.get("/batch-status/{batch_id}")
def batch_status(batch_id: str) -> JSONResponse:
    """
    Verifica el estado del procesamiento del lote
    Retorna información del archivo .parquet generado
    """
    batch_output_dir = BATCH_OUTPUT / batch_id / "parquet_data"

    # Verificar si existe la carpeta de salida
    if not batch_output_dir.exists():
        return JSONResponse(content={
            "batch_id": batch_id,
            "status": "processing",
            "message": "El lote está siendo procesado por PySpark...",
            "parquet_ready": False
        })

    # Verificar si hay archivos .parquet
    parquet_files = list(batch_output_dir.glob("*.parquet"))
    
    if parquet_files:
        return JSONResponse(content={
            "batch_id": batch_id,
            "status": "completed",
            "message": "Procesamiento completado. Resultados listos para descargar.",
            "parquet_ready": True,
            "parquet_files": [f.name for f in parquet_files],
            "output_path": str(batch_output_dir)
        })
    else:
        return JSONResponse(content={
            "batch_id": batch_id,
            "status": "processing",
            "message": "Procesamiento en curso...",
            "parquet_ready": False
        })

@app.get("/download-batch/{batch_id}")
def download_batch(batch_id: str):
    """
    Descarga el archivo .parquet del lote procesado
    """
    batch_output_dir = BATCH_OUTPUT / batch_id / "parquet_data"

    if not batch_output_dir.exists():
        raise HTTPException(status_code=404, detail="Lote no encontrado o aún no procesado")

    # Buscar archivos .parquet
    parquet_files = list(batch_output_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise HTTPException(status_code=404, detail="No hay archivos parquet disponibles")

    # Retornar el primer archivo (o el más reciente)
    parquet_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    
    return FileResponse(
        path=parquet_file,
        media_type="application/octet-stream",
        filename=f"batch_{batch_id}.parquet"
    )

@app.get("/list-batches")
def list_batches() -> JSONResponse:
    """
    Lista todos los lotes procesados
    """
    batches = []
    
    if BATCH_OUTPUT.exists():
        for batch_dir in BATCH_OUTPUT.iterdir():
            if batch_dir.is_dir():
                parquet_dir = batch_dir / "parquet_data"
                parquet_files = list(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
                
                batches.append({
                    "batch_id": batch_dir.name,
                    "status": "completed" if parquet_files else "processing",
                    "parquet_files": len(parquet_files),
                    "created_at": batch_dir.stat().st_ctime
                })

    batches.sort(key=lambda x: x["created_at"], reverse=True)

    return JSONResponse(content={"batches": batches})