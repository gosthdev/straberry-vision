import os
import shutil
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from pyspark.sql import SparkSession, functions as F, types as T
from src.core.config import Config
from src.core.inference import load_model, run_inference_on_image
import matplotlib.pyplot as plt


# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- CONFIGURACIÓN AUTOMÁTICA DE JAVA Y PYTHON ---
# PySpark necesita JAVA_HOME. Si no está configurado, intentamos encontrarlo.
if "JAVA_HOME" not in os.environ:
    possible_java_homes = [
        r"C:\Program Files\Java\jdk-17",
        r"C:\Program Files\Java\jdk-11",
        r"C:\Program Files\Eclipse Adoptium\jdk-17",
    ]
    for java_home in possible_java_homes:
        if os.path.exists(java_home):
            print(f"Configurando JAVA_HOME automáticamente a: {java_home}")
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = os.environ["PATH"] + ";" + os.path.join(java_home, "bin")
            break
    else:
        print("ADVERTENCIA: No se encontró JAVA_HOME ni instalaciones comunes de Java.")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.environ["PYSPARK_NO_DAEMON"] = "true"

BENCHMARK_INPUT_DIR = Path("test/data/benchmark_input")
BENCHMARK_OUTPUT_DIR = Path("test/data/benchmark_output")
MODEL_PATH = Path("src/data/processed/models/best_model.pth")

SPARK_CONF = {
    "master": os.environ.get("SPARK_MASTER", "local[4]"),  # 'local[1]' para 1 core, 'local[*]' para todos
    "driver_memory": os.environ.get("SPARK_DRIVER_MEMORY", "4g"),
    "executor_memory": os.environ.get("SPARK_EXECUTOR_MEMORY", "2g"),
}

# Forzar CPU para igualdad de condiciones (mismo entorno que batch.py)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)
Config.DEVICE = torch.device("cpu")

_model_cache = {}

def setup_data():
    """Prepara datos para el benchmark copiando de batch_incoming"""
    source_dir = Path("test/data/batch_incoming")
    if not source_dir.exists():
        print(f"Directorio fuente {source_dir} no existe. Por favor coloca imágenes .webp ahí.")
        return []
    
    if BENCHMARK_INPUT_DIR.exists():
        shutil.rmtree(BENCHMARK_INPUT_DIR)
    BENCHMARK_INPUT_DIR.mkdir(parents=True)
    
    image_paths = list(source_dir.glob("*.webp"))
    if not image_paths:
        print("No hay imágenes .webp en source_dir")
        return []
        
    print(f"Preparando {len(image_paths)} imágenes para benchmark...")
    final_paths = []
    for p in image_paths:
        dest = BENCHMARK_INPUT_DIR / p.name
        shutil.copy2(p, dest)
        final_paths.append(str(dest))
        
    return final_paths

# --- IMPLEMENTACIÓN SPARK ---

def build_session():
    print(f"Iniciando Spark con: Master={SPARK_CONF['master']}, DriverMem={SPARK_CONF['driver_memory']}")
    return (
        SparkSession.builder
        .appName("StrawberryBenchmark")
        .master(SPARK_CONF["master"])
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", SPARK_CONF["driver_memory"])
        .config("spark.executor.memory", SPARK_CONF["executor_memory"])
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

def run_spark_benchmark(image_paths):
    print("\n--- Iniciando Benchmark Spark ---")
    start_time = time.time()
    
    spark = build_session()
    
    spark_output = BENCHMARK_OUTPUT_DIR / "spark"
    if spark_output.exists():
        shutil.rmtree(spark_output)
        
    # Crear DataFrame con los paths
    df = spark.createDataFrame([(p,) for p in image_paths], ["path"])
    
    # Reparticionar para asegurar paralelismo (ajustar según CPU cores)
    num_partitions = max(4, os.cpu_count() or 2)
    df = df.repartition(num_partitions)
    
    # Ejecutar inferencia distribuida
    result = df.withColumn("detections", infer_udf(F.col("path")))
    
    # Acción de escritura para materializar resultados
    result.write.mode("overwrite").parquet(str(spark_output))
    
    spark.stop()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Spark completado en: {duration:.4f} segundos")
    return duration

# --- IMPLEMENTACIÓN SECUENCIAL ---

def run_sequential_benchmark(image_paths):
    print("\n--- Iniciando Benchmark Secuencial ---")
    start_time = time.time()
    
    # Cargar modelo una vez
    model = load_model(str(MODEL_PATH.resolve()))
    
    results = []
    
    # Procesamiento en serie
    for path in image_paths:
        preds = run_inference_on_image(model, path, conf_threshold=Config.CONF_THRESHOLD)
        results.append({
            "path": path,
            "detections": preds
        })
    
    # Guardar resultados
    seq_output = BENCHMARK_OUTPUT_DIR / "sequential"
    seq_output.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_json(seq_output / "results.json")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Secuencial completado en: {duration:.4f} segundos")
    return duration



def plot_results(seq_time, spark_time, num_images):
    """Genera un gráfico comparativo de los tiempos de ejecución"""
    methods = ['Secuencial', 'Spark']
    times = [seq_time, spark_time]
    colors = ['#ff9999', '#66b3ff']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=colors)
    
    # Añadir etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')

    plt.title(f'Comparación de Rendimiento: Procesamiento de {num_images} Imágenes')
    plt.ylabel('Tiempo Total (segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir información de velocidad por imagen
    plt.text(0, seq_time/2, f'{seq_time/num_images:.3f} s/img', ha='center', color='black', fontweight='bold')
    plt.text(1, spark_time/2, f'{spark_time/num_images:.3f} s/img', ha='center', color='black', fontweight='bold')

    output_path = BENCHMARK_OUTPUT_DIR / "benchmark_comparison.png"
    plt.savefig(output_path)
    print(f"\nGráfico guardado en: {output_path}")

def main():
    print("=== BENCHMARK DE RENDIMIENTO: SPARK vs SECUENCIAL ===")
    image_paths = setup_data()
    
    if not image_paths:
        return

    # 1. Ejecutar Secuencial
    seq_time = run_sequential_benchmark(image_paths)
    
    # 2. Ejecutar Spark
    spark_time = run_spark_benchmark(image_paths)
    
    print("\n" + "="*40)
    print("RESULTADOS FINALES")
    print("="*40)
    print(f"Imágenes procesadas: {len(image_paths)}")
    print(f"Tiempo Secuencial:   {seq_time:.4f} s  ({seq_time/len(image_paths):.4f} s/img)")
    print(f"Tiempo Spark:        {spark_time:.4f} s  ({spark_time/len(image_paths):.4f} s/img)")
    print("-" * 40)
    
    if spark_time < seq_time:
        speedup = seq_time / spark_time
        print(f"CONCLUSIÓN: Spark fue {speedup:.2f}x más RÁPIDO.")
    else:
        slowdown = spark_time / seq_time
        print(f"CONCLUSIÓN: Spark fue {slowdown:.2f}x más LENTO.")
        print("Nota: Spark tiene un overhead de inicio alto. Para pocas imágenes, es normal que sea más lento.")

    # Generar gráfico
    plot_results(seq_time, spark_time, len(image_paths))

if __name__ == "__main__":
    main()
