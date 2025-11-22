# straberry-vision

## Java runtime setup

This project targets Apache Spark 4.x, which is only compatible with Java 17. If your system default is a newer JDK (for example, Arch Linux ships Java 25), set `JAVA_HOME` before running any Spark scripts:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
```

To make the change persistent on Arch-based systems you can also run (requires sudo):

Run the batch pipeline or inspection commands from the virtual environment **after** exporting these variables so PySpark picks up the correct JDK.

PYTHONPATH=$(pwd) spark-submit \
  --master local[1] \
  --conf spark.driver.memory=2g \
  --conf spark.executor.memory=2g \
  --conf spark.executor.cores=1 \
  src/core/batch.py
```

Al terminar, Spark deja un único parquet (`part-*.parquet`) y un CSV plano `batch_outputs.csv` en `test/data/batch_outputs/` listos para inspección.

Si necesitas exportaciones adicionales (por ejemplo, sin ordenar o hacia otra ruta), puedes usar el script auxiliar:

```bash
source env/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python src/utils/export_batch_outputs.py --out test/data/batch_outputs_flat
```

El archivo generado queda en `test/data/batch_outputs_flat/part-*.csv`.

### Ejecución automática cuando llegan imágenes

Si quieres que el pipeline se ejecute solo cada vez que aparezcan imágenes nuevas en `test/data/batch_incoming/`, puedes usar el nuevo watcher:

```bash
source env/bin/activate
export BATCH_INPUT_DIR=/ruta/compartida/batch_incoming  # opcional
export SPARK_MASTER_URL=spark://<host-master>:7077      # opcional
python -m src.app.batch_watcher --run-on-start
```

El watcher utiliza `watchdog` para monitorear el directorio. Cuando detecta nuevos `.webp` (puedes ajustar la espera con `--debounce`), dispara `spark-submit` con los parámetros configurados por variables de entorno:

- `SPARK_SUBMIT_CMD`: comando para lanzar Spark (por defecto `spark-submit`).
- `SPARK_SUBMIT_ARGS`: flags extra (`--conf`, `--deploy-mode`, etc.).
- `BATCH_MAX_IMAGES`: tamaño de lote del pipeline.

Presiona `Ctrl+C` para detener el proceso.

## Orquestación con Docker

El repositorio incluye un `docker-compose.yml` que levanta todo el stack:

- `spark-master`: nodo master de Spark con UI en `http://localhost:8080`.
- `spark-worker`: worker conectado al master (puedes escalar más replicas).
- `batch-watcher`: contenedor que monta el repositorio, monitorea la carpeta compartida y lanza `spark-submit` cuando detecta imágenes nuevas.

### Preparación

1. Crea un directorio compartido en tu host para intercambiar archivos con los contenedores:
  ```bash
  mkdir -p shared/batch_incoming shared/batch_outputs shared/batch_outputs_staging
  ```
2. Copia el modelo `best_model.pth` al contenedor (ya está en `src/data/processed/models/`); si tienes uno actualizado, emplázalo ahí antes de construir la imagen.
3. Construye las imágenes y levanta los servicios:
  ```bash
  docker compose build
  docker compose up -d
  ```

El volumen `shared-data` se monta en `/data` dentro de los contenedores. Coloca tus imágenes `.webp` en `shared/batch_incoming/` y el watcher disparará el pipeline. Los resultados aparecen en `shared/batch_outputs/`.

### Variables relevantes

Puedes sobreescribir las variables del watcher al ejecutar `docker compose`, por ejemplo:

```bash
export BATCH_WATCH_DEBOUNCE=20
export BATCH_WATCH_MIN_FILES=10
docker compose up -d batch-watcher
```

O modifica las entradas en `docker-compose.yml` según tus necesidades (`BATCH_MAX_IMAGES`, `SPARK_SUBMIT_ARGS`, etc.).

### Escalar workers

Para añadir más workers solo duplica el servicio `spark-worker` o utiliza:

```bash
docker compose up -d --scale spark-worker=3
```

Spark balanceará la carga automáticamente; el batch seguirá compactando el resultado final en un único parquet y CSV.
