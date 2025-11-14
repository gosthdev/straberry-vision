# Straberry Vision

Proyecto de visión por computadora para estimar el estado de maduración de fresas. La solución combina entrenamiento supervisado con PyTorch, inferencia en tiempo real vía FastAPI y un pipeline batch con Spark para procesar grandes volúmenes de imágenes.

## Arquitectura principal

- **Backbone ResNet34 + FPN**: se utilizan mapas P3/P4/P5 con detección multiescala para mejorar recall en frutos de distintos tamaños.
- **Anchors dinámicos**: al iniciar el módulo `Config`, se recalculan 9 anchors normalizados mediante k-means sobre las anotaciones COCO disponibles, ajustando automáticamente el detector a nuevos datasets.
- **Cabezas de detección YOLO-like**: tres cabezas compartidas con activaciones SiLU que predicen `bbox`, confianza y clases por escala.
- **Pérdida multiescala**: asignación de targets respetando stride por nivel, con términos BCE y IoU para estabilizar entrenamiento.

## Pipeline batch

- Spark en modo local procesa carpetas de imágenes, limitando el número por lote y moviendo archivos ya procesados a un directorio `processed/`.
- Configurable mediante `Config` para ajustar rutas, límites y retención.
- Útil para reprocesar históricos sin saturar la GPU principal.

## Inferencia

- API REST construida con FastAPI (`app/main.py`) que expone endpoints para subir imágenes y obtener detecciones.
- Utilidades en `src/core/inference.py` para probar un checkpoint en una imagen o en carpetas completas, generando métricas y visualizaciones.
- Decodificación de predicciones multiescala con umbral configurable (`Config.CONF_THRESHOLD`).

## Estructura relevante

```
src/
	core/
		architecture.py   # Modelo SGSNet con ResNet34 + FPN
		config.py         # Hiperparámetros, paths, anchors dinámicos
		dataset.py        # Dataset y transforms Albumentations
		trainer.py        # Loop de entrenamiento y validación
		inference.py      # Utilidades de inferencia y visualización
	utils/
		analitjson.py     # Análisis de anotaciones
		validation.py     # Métricas adicionales
app/
	main.py            # FastAPI para inferencia online
batch/
	(scripts spark)    # Procesamiento por lotes
```

## Requisitos

- Python 3.13 (virtualenv en `env/`)
- PyTorch + torchvision (GPU recomendada, soporte CPU disponible)
- FastAPI, Uvicorn para el servicio REST
- Spark (pyspark, pyarrow) para el pipeline batch
- Albumentations, OpenCV, NumPy, SciPy

Instalación rápida:

```powershell
python -m venv env
env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Flujo básico

1. **Preparar datos**: colocar imágenes en `src/data/raw/images` y anotaciones COCO en `src/data/processed/coco_out/annotations.json`.
2. **Entrenar**: ejecutar `python src/core/trainer.py` (revisa `Config` para ajustar hiperparámetros y rutas de salida).
3. **Inferir**:
	 - Imagen única: `python -m src.core.inference --model path_al_checkpoint --image path_a_imagen`.
	 - Carpeta: `python test/evaluate_trained_model.py --weights path_al_checkpoint --folder path_imagenes`.
	 - API: `uvicorn app.main:app --reload --port 8000`.
4. **Procesamiento batch**: levantar el script Spark configurando Java y `pyspark` (ver instrucciones internas del proyecto).

## Consideraciones

- El recálculo de anchors ocurre al importar `Config`. Si se actualizan las anotaciones, vuelve a ejecutar los scripts para que los anchors se adapten.
- El pipeline admite entrenamiento en GPU de alta capacidad pero conserva compatibilidad CPU para pruebas.
- Ajusta `Config.CONF_THRESHOLD` y `Config.IOU_THRESHOLD` según el balance precisión/recall deseado.

## Próximos pasos sugeridos

- Integrar métricas de validación en `trainer.py` usando las nuevas salidas multiescala.
- Automatizar el pipeline batch/inferencia en CI/CD con GitHub Actions.
- Evaluar retraining con mayor diversidad de clases negativas para reducir falsos positivos.
