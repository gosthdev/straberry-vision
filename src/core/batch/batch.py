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
from pyspark import SparkFiles
from src.core.model.config import Config
from src.core.model.inference import load_model, run_inference_on_image

# Forzar ejecución en CPU para reducir consumo y evitar OOM en GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)
Config.DEVICE = torch.device("cpu")

MODEL_PATH = Path(os.environ.get("BATCH_MODEL_PATH", "src/data/processed/models/best_model.pth"))
INPUT_DIR = Path(os.environ.get("BATCH_INPUT_DIR", "test/data/batch_incoming"))
OUTPUT_DIR = Path(os.environ.get("BATCH_OUTPUT_DIR", "test/data/batch_outputs"))
DEFAULT_STAGING = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_staging"
STAGING_DIR = Path(os.environ.get("BATCH_STAGING_DIR", str(DEFAULT_STAGING)))
MAX_IMAGES = int(os.environ.get("BATCH_MAX_IMAGES", "150"))
SPARK_MASTER = os.environ.get("SPARK_MASTER_URL", "local[*]")

_model_cache = {}

def build_session():
    return (
        SparkSession.builder
        .appName("StrawberryBatch")
        .master(SPARK_MASTER)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Asegurar que los workers NO intenten usar GPU (evita OOM y conflictos)
        .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "")
        # Evitar reutilización de workers para prevenir SegFaults con PyTorch
        .config("spark.python.worker.reuse", "false")
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
    # Importar y configurar torch localmente en el worker para asegurar aislamiento
    import torch
    torch.set_num_threads(1)

    if "model" not in _model_cache:

        try:
            model_file = SparkFiles.get(MODEL_PATH.name)
            _model_cache["model"] = load_model(model_file)
        except Exception:
            _model_cache["model"] = load_model(str(MODEL_PATH.resolve()))
            
    preds = run_inference_on_image(_model_cache["model"], path,
                                   conf_threshold=Config.CONF_THRESHOLD)
    return preds

def run():
    spark = build_session()
    
    # Distribuir el archivo del modelo a todos los nodos del clúster (o workers locales)
    if MODEL_PATH.exists():
        spark.sparkContext.addFile(str(MODEL_PATH.resolve()))
    else:
        print(f"Advertencia: No se encontró el modelo en {MODEL_PATH}, la inferencia podría fallar.")
    
    # Configuración de directorios
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Carga de datos 
    df = (spark.read.format("binaryFile")
          .option("pathGlobFilter", "*.webp")
          .option("recursiveFileLookup", "true")
          .load(str(INPUT_DIR))
          .select("path"))
    # Verificar si hay datos
    if df.rdd.isEmpty():
        print("No hay imágenes para procesar.")
        spark.stop()
        return
    # 2. Reparticionar para asegurar paralelismo real
    # Usamos el número de CPUs disponibles para maximizar el uso de recursos
    num_cores = os.cpu_count() or 2
    df = df.repartition(num_cores)
    # 3. Inferencia Distribuida
    # Ejecutamos la UDF en paralelo sobre las particiones
    result_df = df.withColumn("detections", infer_udf(F.col("path")))
    result_df.cache()
    # 4. Escritura de resultados crudos (Parquet)
    parquet_output = OUTPUT_DIR / "parquet_data"
    (result_df.write
        .mode("overwrite")
        .parquet(str(parquet_output)))
    # 5. Limpieza de archivos de entrada (Post-procesamiento)
    # Solo borramos los archivos si el proceso de Spark terminó correctamente
    print("Limpiando archivos procesados...")
    
    def delete_files_partition(iterator):
        """Función que se ejecuta en los workers para borrar archivos en paralelo"""
        for row in iterator:
            try:
                p = Path(row.path)
                if p.exists():
                    p.unlink()
            except Exception as e:
                # Esto saldrá en los logs del executor (stderr)
                print(f"Error borrando {row.path}: {e}")

    # Usamos foreachPartition para que cada worker borre sus archivos
    # Esto evita traer millones de rutas al driver con collect() (evita OOM)
    df.select("path").foreachPartition(delete_files_partition)

    spark.stop()
    print("Procesamiento completado exitosamente.")

if __name__ == "__main__":
    run()