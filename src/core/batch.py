import os
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path para spark-submit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pyspark.sql import SparkSession, functions as F, types as T
from src.core.config import Config
from src.core.inference import load_model, run_inference_on_image

# Forzar ejecución en CPU para reducir consumo y evitar OOM en GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)
Config.DEVICE = torch.device("cpu")

MODEL_PATH = Path("src/data/processed/models/best_model.pth")
INPUT_DIR = Path("test/data/batch_incoming")
OUTPUT_DIR = Path("test/data/batch_outputs")
PROCESSED_DIR = Path("test/data/batch_idk")
MAX_IMAGES = int(os.environ.get("BATCH_MAX_IMAGES", "250"))

_model_cache = {}

def build_session():
    return (SparkSession.builder
            .appName("StrawberryBatch")
            .master("local[*]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate())

@F.udf(returnType=T.ArrayType(T.StructType([
    T.StructField("label", T.StringType()),
    T.StructField("score", T.FloatType()),
    T.StructField("xmin", T.FloatType()),
    T.StructField("ymin", T.FloatType()),
    T.StructField("xmax", T.FloatType()),
    T.StructField("ymax", T.FloatType()),
])))
def infer_udf(path):
    if "model" not in _model_cache:
        _model_cache["model"] = load_model(str(MODEL_PATH.resolve()))
    preds = run_inference_on_image(_model_cache["model"], path,
                                   conf_threshold=Config.CONF_THRESHOLD)
    return preds

def run():
    spark = build_session()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = (spark.read.format("binaryFile")
          .option("pathGlobFilter", "*.webp")
          .load(str(INPUT_DIR))
        .select("path"))

    # Limitar el número de particiones para reducir tareas concurrentes
    df = df.coalesce(1)

    if MAX_IMAGES > 0:
        df = df.limit(MAX_IMAGES)

    if df.rdd.isEmpty():
        print("No hay imágenes para procesar.")
        spark.stop()
        return

    result = df.withColumn("detections", infer_udf(F.col("path")))
    result.write.mode("overwrite").parquet(str(OUTPUT_DIR))

    processed_paths = [Path(row.path) for row in df.select("path").collect()]
    for path in processed_paths:
        if not path.exists():
            continue
        target = PROCESSED_DIR / path.name
        counter = 1
        while target.exists():
            target = PROCESSED_DIR / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        path.replace(target)
    spark.stop()

if __name__ == "__main__":
    run()