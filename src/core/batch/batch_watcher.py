"""Watcher de archivos que ejecuta el batch de Spark al detectar nuevas imágenes."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSIONS = {".webp"}
DEFAULT_DEBOUNCE_SECONDS = 10


class BatchRunner:
    """Encapsula la lógica para lanzar el batch de Spark de forma segura."""

    def __init__(
        self,
        input_dir: Path,
        command: List[str],
        env: Optional[dict] = None,
        min_files: int = 1,
    ) -> None:
        self.input_dir = input_dir
        self.command = command
        self.env = env or os.environ.copy()
        self.min_files = min_files
        self._run_lock = threading.Lock()

    def has_pending_files(self) -> bool:
        count = sum(1 for _ in self._iter_candidate_files())
        logging.debug("Archivos pendientes detectados: %s", count)
        return count >= self.min_files

    def _iter_candidate_files(self) -> Iterable[Path]:
        for pattern in DEFAULT_EXTENSIONS:
            yield from self.input_dir.glob(f"*{pattern}")

    def run_batch(self) -> None:
        if not self.has_pending_files():
            logging.info("Ejecución omitida: no hay archivos compatibles en %s", self.input_dir)
            return

        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            logging.info("Ya hay un procesamiento en curso; se omite el disparo")
            return

        try:
            logging.info("Lanzando batch de Spark: %s", " ".join(self.command))
            start = time.time()
            completed = subprocess.run(self.command, env=self.env, check=False)
            duration = time.time() - start

            if completed.returncode == 0:
                logging.info("Batch finalizado correctamente en %.1fs", duration)
            else:
                logging.error(
                    "Batch falló (código %s) tras %.1fs",
                    completed.returncode,
                    duration,
                )
        except Exception:  # pragma: no cover - defensive
            logging.exception("Error inesperado al ejecutar el batch")
        finally:
            self._run_lock.release()


class DebouncedTrigger(FileSystemEventHandler):
    """Programa la ejecución del batch tras un periodo de calma en los eventos."""

    def __init__(
        self,
        runner: BatchRunner,
        debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS,
        extensions: Optional[Iterable[str]] = None,
    ) -> None:
        self.runner = runner
        self.debounce_seconds = debounce_seconds
        self.extensions = set(ext.lower() for ext in (extensions or DEFAULT_EXTENSIONS))
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:  # pragma: no cover - I/O heavy
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in self.extensions:
            return

        logging.debug("Cambio detectado en %s (%s)", path, event.event_type)
        self._schedule_trigger()

    def _schedule_trigger(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.runner.run_batch)
            self._timer.daemon = True
            self._timer.start()
            logging.info(
                "Batch programado dentro de %ss (espera de consolidación)",
                self.debounce_seconds,
            )


def build_default_command(project_root: Path, args: argparse.Namespace) -> List[str]:
    spark_submit = args.spark_submit or os.environ.get("SPARK_SUBMIT_CMD", "spark-submit")
    master_url = args.master or os.environ.get("SPARK_MASTER_URL", "local[*]")
    extra_args = args.extra_args or os.environ.get("SPARK_SUBMIT_ARGS", "")
    app_path = args.app or project_root / "src" / "core" / "batch.py"

    command = [spark_submit, "--master", master_url]
    if extra_args:
        command.extend(shlex.split(extra_args))
    if args.executor_python and not any(
        "spark.executorEnv.PYSPARK_PYTHON" in token for token in command
    ):
        command.extend([
            "--conf",
            f"spark.executorEnv.PYSPARK_PYTHON={args.executor_python}",
        ])
    command.append(str(app_path))
    return command


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline batch cuando hay archivos nuevos")
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("BATCH_INPUT_DIR", "test/data/batch_incoming"),
    help="Directorio a monitorear en busca de archivos nuevos",
    )
    parser.add_argument(
        "--spark-submit",
    help="Ruta al ejecutable spark-submit (por defecto SPARK_SUBMIT_CMD o 'spark-submit')",
    )
    parser.add_argument(
        "--master",
    help="URL del master de Spark (por defecto SPARK_MASTER_URL o local[*])",
    )
    parser.add_argument(
        "--app",
    help="Punto de entrada del batch (por defecto src/core/batch.py)",
    )
    parser.add_argument(
        "--extra-args",
    help="Argumentos adicionales que se pasan tal cual a spark-submit (cadena entre comillas)",
    )
    parser.add_argument(
        "--executor-python",
        default=os.environ.get("PYSPARK_EXECUTOR_PYTHON")
        or os.environ.get("BATCH_EXECUTOR_PYTHON"),
    help="Ruta al intérprete de Python para los workers (propaga spark.executorEnv.PYSPARK_PYTHON)",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=int(os.environ.get("BATCH_WATCH_DEBOUNCE", DEFAULT_DEBOUNCE_SECONDS)),
    help="Segundos de espera tras el último evento antes de disparar el batch",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=int(os.environ.get("BATCH_WATCH_MIN_FILES", "1")),
    help="Número mínimo de archivos pendientes para lanzar el batch",
    )
    parser.add_argument(
        "--run-on-start",
        action="store_true",
    help="Ejecuta el batch de inmediato si hay archivos pendientes al iniciar",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("BATCH_WATCH_LOG_LEVEL", "INFO"),
    help="Nivel de logging (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_dir = Path(args.input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if args.executor_python:
        env.setdefault("PYSPARK_EXECUTOR_PYTHON", args.executor_python)

    command = build_default_command(PROJECT_ROOT, args)

    runner = BatchRunner(input_dir=input_dir, command=command, env=env, min_files=args.min_files)
    handler = DebouncedTrigger(runner=runner, debounce_seconds=args.debounce)

    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=False)

    logging.info("Monitoreando %s en busca de archivos nuevos...", input_dir)
    logging.info("Comando configurado: %s", " ".join(command))

    if args.run_on_start:
        logging.info("Ejecución inicial solicitada mediante --run-on-start")
        runner.run_batch()

    observer.start()
    try:
        while True:  # pragma: no cover - long running loop
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Deteniendo watcher...")
    finally:
        observer.stop()
        observer.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
