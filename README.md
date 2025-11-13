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
