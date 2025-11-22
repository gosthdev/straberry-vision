import os
import shutil
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

MODEL_PATH = Path(os.environ.get("BATCH_MODEL_PATH", "src/data/processed/models/best_model.pth"))
INPUT_DIR = Path(os.environ.get("BATCH_INPUT_DIR", "test/data/batch_incoming"))
OUTPUT_DIR = Path(os.environ.get("BATCH_OUTPUT_DIR", "test/data/batch_outputs"))
DEFAULT_STAGING = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_staging"
STAGING_DIR = Path(os.environ.get("BATCH_STAGING_DIR", str(DEFAULT_STAGING)))
CSV_FILENAME = "batch_outputs.csv"
MAX_IMAGES = int(os.environ.get("BATCH_MAX_IMAGES", "150"))
SPARK_MASTER = os.environ.get("SPARK_MASTER_URL", "local[*]")

_model_cache = {}

def build_session():
    return (
        SparkSession.builder
        .appName("StrawberryBatch")
        .master(SPARK_MASTER)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / CSV_FILENAME
    if csv_path.exists():
        csv_path.unlink()

    image_paths = sorted(INPUT_DIR.glob("*.webp"))

    if not image_paths:
        print("No hay imágenes para procesar.")
        spark.stop()
        return

    batch_size = MAX_IMAGES if MAX_IMAGES > 0 else len(image_paths)
    total_images = len(image_paths)

    for start in range(0, total_images, batch_size):
        chunk = image_paths[start:start + batch_size]
        chunk_as_str = [str(path) for path in chunk]

        df = (spark.read.format("binaryFile")
              .load(chunk_as_str)
              .select("path"))

        df = df.coalesce(1)

        result = df.withColumn("detections", infer_udf(F.col("path")))
        result.write.mode("append").parquet(str(STAGING_DIR))

        for path in chunk:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    final_df = spark.read.parquet(str(STAGING_DIR))
    (final_df.coalesce(1)
             .write
             .mode("overwrite")
             .parquet(str(OUTPUT_DIR)))

    detections_df = (
        final_df
        .select(F.col("path"), F.explode("detections").alias("det"))
        .select(
            F.col("path"),
            F.col("det.label").alias("label"),
            F.col("det.score").alias("score"),
            F.col("det.xmin").alias("xmin"),
            F.col("det.ymin").alias("ymin"),
            F.col("det.xmax").alias("xmax"),
            F.col("det.ymax").alias("ymax"),
        )
        .orderBy("path", F.desc("score"))
    )

    csv_temp_dir = OUTPUT_DIR / "_csv_temp"
    if csv_temp_dir.exists():
        shutil.rmtree(csv_temp_dir)

    detections_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_temp_dir))

    part_file = next(csv_temp_dir.glob("part-*.csv"), None)
    if part_file is not None:
        shutil.move(str(part_file), str(csv_path))
    else:
        header = ",".join(field.name for field in detections_df.schema)
        csv_path.write_text(header + "\n")

    shutil.rmtree(csv_temp_dir, ignore_errors=True)
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    spark.stop()

if __name__ == "__main__":
    run()