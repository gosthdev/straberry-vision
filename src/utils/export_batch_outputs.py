import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


def build_spark_session(app_name: str = "ExportBatchOutputs") -> SparkSession:
    return (
        SparkSession.builder
        .master("local[1]")
        .appName(app_name)
        .getOrCreate()
    )


def export_parquet(parquet_path: Path, output_path: Path, sort_column: str = "path") -> None:
    spark = build_spark_session()

    try:
        df = spark.read.parquet(str(parquet_path))
    except Exception:
        spark.stop()
        raise

    detections = (
        df
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
    )

    if sort_column:
        detections = detections.orderBy(sort_column, F.desc("score"))

    (
        detections
        .coalesce(1)  # un único archivo para inspección manual
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(output_path))
    )

    spark.stop()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte el parquet de resultados de inferencia por lotes en un CSV "
            "aplanado y legible"
        )
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("test/data/batch_outputs"),
        help="Directorio o archivo parquet generado por Spark",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("test/data/batch_outputs_csv"),
        help="Directorio destino para el CSV (se sobrescribe)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="No ordenar por ruta y score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"No se encontró el parquet: {args.parquet}")

    args.out.mkdir(parents=True, exist_ok=True)

    sort_column = "" if args.no_sort else "path"
    export_parquet(args.parquet, args.out, sort_column=sort_column)
    print(
        f"CSV generado en {args.out}. Puedes abrir el archivo con cualquier hoja de cálculo "
        "(Spark colocará un archivo \"part-*.csv\" con los datos)."
    )

if __name__ == "__main__":
    main()
