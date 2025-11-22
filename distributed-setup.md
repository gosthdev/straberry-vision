# Guía completa: Master y Workers en equipos separados

Esta guía describe cómo desplegar el pipeline batch de `strawberry-vision` con Apache Spark en dos (o más) computadoras físicas: un nodo **master** dedicado a coordinar el cluster y al watcher automático, y uno o varios nodos **worker** que ejecutan las tareas de inferencia. Además, se explica cómo compartir los datos mediante Samba para que todos los nodos vean las mismas imágenes y resultados.

## 1. Requisitos previos

| Recurso                | Master (Linux) | Worker (Linux) | Worker (Windows) |
|------------------------|:--------------:|:--------------:|:----------------:|
| Linux (Ubuntu, Debian, Arch, etc.) | ✅ | ✅ | Opcional |
| Windows 10/11 (PowerShell/WSL) | ❌ | Opcional | ✅ |
| Java 17 (JDK)          | ✅ | ✅ | ✅ |
| Python 3.11+ (para watcher) | ✅ | Opcional | Opcional |
| Apache Spark 4.0.*     | ✅ | ✅ | ✅ |
| Acceso a red (LAN o VPN) | ✅ | ✅ | ✅ |
| Carpeta compartida (Samba/CIFS) | ✅ | ✅ | ✅ |

### Instalación rápida (Linux)

```bash
# En ambos equipos
sudo pacman -S jdk17-openjdk python-virtualenv git        # ejemplo en Arch
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"

# Descarga Spark
curl -O https://downloads.apache.org/spark/spark-4.0.0/spark-4.0.0-bin-hadoop3.tgz
sudo tar -xf spark-4.0.0-bin-hadoop3.tgz -C /opt
export SPARK_HOME=/opt/spark-4.0.0-bin-hadoop3
export PATH="$SPARK_HOME/bin:$PATH"
```

### Instalación rápida (Worker Windows)

1. Instala Java 17 (Zulu OpenJDK o Temurin) y agrega la ruta al PATH.
2. Descarga Spark 4.0.0 (prebuilt Hadoop3) y descomprime en `C:\spark-4.0.0-bin-hadoop3`.
3. Configura variables de entorno (PowerShell, ejecutar como administrador):
   ```powershell
   setx JAVA_HOME "C:\Program Files\Java\jdk-17"
   setx SPARK_HOME "C:\spark-4.0.0-bin-hadoop3"
   setx HADOOP_HOME "C:\spark-4.0.0-bin-hadoop3"
   setx PATH "%PATH%;%SPARK_HOME%\bin;%JAVA_HOME%\bin"
   ```
4. Descarga `winutils.exe` compatible con Hadoop 3.x (por ejemplo, desde https://github.com/steveloughran/winutils) y colócalo en `%HADOOP_HOME%\bin`.
5. Reinicia la terminal para que las variables surtan efecto.

## 2. Compartir datos con Samba

1. **En el equipo que almacenará las imágenes** (puede ser el master):
   ```bash
   sudo mkdir -p /srv/strawberry/{batch_incoming,batch_outputs,batch_outputs_staging,models}
   sudo chown -R usuario:grupo /srv/strawberry
   ```
2. Configura Samba (`/etc/samba/smb.conf`):
   ```ini
   [strawberry]
   path = /srv/strawberry
   writable = yes
   browseable = yes
   guest ok = no
   valid users = usuario
   force create mode = 0664
   force directory mode = 0775
   ```
3. Reinicia el servicio y crea la contraseña:
   ```bash
   sudo systemctl restart smb nmb
   sudo smbpasswd -a usuario
   ```
4. **En cada nodo Linux (master y workers)** monta la carpeta:
   ```bash
   sudo mkdir -p /mnt/strawberry
   sudo mount -t cifs //<host_samba>/strawberry /mnt/strawberry \
     -o username=usuario,password=secreta,uid=$(id -u),gid=$(id -g),file_mode=0664,dir_mode=0775
   ```
   Para montaje automático, añade la entrada en `/etc/fstab`.

5. **En workers Windows** asigna la unidad de red (PowerShell):
    ```powershell
    New-PSDrive -Name "S" -PSProvider FileSystem -Root "\\\<host_samba>\strawberry" \
       -Persist -Credential (Get-Credential)
    ```
    La carpeta compartida quedará disponible como `S:\batch_incoming`, `S:\batch_outputs`, etc.

El watcher y Spark leerán imágenes desde `/mnt/strawberry/batch_incoming` y depositarán los resultados en `/mnt/strawberry/batch_outputs`.

## 3. Configuración del nodo Master

### 3.1 Preparar el proyecto y entorno
```bash
# En el master
git clone https://github.com/<tu-usuario>/strawberry-vision.git
cd strawberry-vision
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Variables de entorno clave
Añade a tu `.bashrc` o exporta antes de arrancar servicios:
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export SPARK_HOME=/opt/spark
export PATH="$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH"

export BATCH_INPUT_DIR=/mnt/strawberry/batch_incoming
export BATCH_OUTPUT_DIR=/mnt/strawberry/batch_outputs
export BATCH_STAGING_DIR=/mnt/strawberry/batch_outputs_staging
export BATCH_MODEL_PATH=/mnt/strawberry/models/best_model.pth
export BATCH_MAX_IMAGES=150
export SPARK_MASTER_URL=spark://<ip_master>:7077
export SPARK_SUBMIT_CMD=$SPARK_HOME/bin/spark-submit
export SPARK_SUBMIT_ARGS="--deploy-mode client --conf spark.executor.memory=3g --conf spark.executor.cores=2"
```

### 3.3 Arrancar el master de Spark
```bash
$SPARK_HOME/sbin/start-master.sh --host <ip_master> --port 7077
```
- UI disponible en `http://<ip_master>:8080`.
- Apunta la URL que imprime (`spark://<ip_master>:7077`); la necesitarán los workers y el watcher.

### 3.4 Ejecutar el watcher automático
```bash
source env/bin/activate
python -m src.app.batch_watcher --run-on-start --debounce 15 --min-files 5
# Si tus workers usan un intérprete específico (por ejemplo, un venv en Windows),
# añade --executor-python con la ruta remota que deben usar los ejecutores:
# python -m src.app.batch_watcher --run-on-start --executor-python "C:/ruta/a/env/Scripts/python.exe"
```
- El watcher revisa la carpeta compartida y lanza `spark-submit` cada vez que detecta nuevas imágenes.
- Ajusta `--debounce` y `--min-files` según tus necesidades.

#### Servicio systemd (opcional)
Crea `/etc/systemd/system/strawberry-watcher.service`:
```ini
[Unit]
Description=Watcher batch Strawberry
After=network.target

[Service]
User=tu_usuario
WorkingDirectory=/home/tu_usuario/strawberry-vision
Environment=JAVA_HOME=/usr/lib/jvm/java-17-openjdk
Environment=SPARK_HOME=/opt/spark-4.0.0-bin-hadoop3
Environment=SPARK_MASTER_URL=spark://<ip_master>:7077
Environment=BATCH_INPUT_DIR=/mnt/strawberry/batch_incoming
Environment=BATCH_OUTPUT_DIR=/mnt/strawberry/batch_outputs
Environment=BATCH_STAGING_DIR=/mnt/strawberry/batch_outputs_staging
Environment=BATCH_MODEL_PATH=/mnt/strawberry/models/best_model.pth
Environment=BATCH_MAX_IMAGES=150
Environment=SPARK_SUBMIT_CMD=/opt/spark-4.0.0-bin-hadoop3/bin/spark-submit
Environment=SPARK_SUBMIT_ARGS=--deploy-mode client --conf spark.executor.memory=3g --conf spark.executor.cores=2
ExecStart=/home/tu_usuario/strawberry-vision/env/bin/python -m src.app.batch_watcher --run-on-start
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Activa el servicio: `sudo systemctl enable --now strawberry-watcher`.

## 4. Configuración de los nodos Worker

### 4.1 Worker Linux

1. **Instala Java, Spark y monta Samba** igual que en el master.
2. **Conecta el worker al master**:
   ```bash
   $SPARK_HOME/sbin/start-worker.sh spark://<ip_master>:7077 \
     --host <ip_worker> \
     --webui-port 8081 \
     --cores 2 \
     --memory 4G
   ```
3. **Si tus dependencias viven en un `virtualenv` o Conda**:
   ```bash
   python -m venv /opt/strawberry-env
   source /opt/strawberry-env/bin/activate
   pip install -r /ruta/al/proyecto/requirements.txt
   export PYSPARK_PYTHON=/opt/strawberry-env/bin/python
   ```
   Coloca ese `export` en `$SPARK_HOME/conf/spark-env.sh` o en el servicio systemd del worker
   para que Spark propague la misma ruta a todos los ejecutores Linux.
4. Verifica en la UI del master que el worker aparece conectado.

### 4.2 Worker Windows

1. Abre PowerShell (preferentemente como administrador).
2. Asegúrate de que la unidad compartida está montada (ejemplo `S:`) y de que `JAVA_HOME`, `SPARK_HOME` y `PATH` están configurados.
3. Ejecuta el script `start-worker.cmd` que viene con Spark:
   ```powershell
   cd $env:SPARK_HOME\sbin
   .\start-worker.cmd spark://<ip_master>:7077 --cores 2 --memory 4G --webui-port 8081
   ```
4. **Para usar un entorno virtual de Python** (venv, Conda, etc.):
   ```powershell
   # Crear/activar el entorno una sola vez
   python -m venv C:\strawberry-env
   C:\strawberry-env\Scripts\Activate.ps1
   pip install -r S:\ruta\al\proyecto\requirements.txt

   # Hacer persistente la ruta para Spark
   setx PYSPARK_PYTHON "C:\\strawberry-env\\Scripts\\python.exe"
   # o arranca el worker con el flag
   .\start-worker.cmd spark://<ip_master>:7077 --cores 2 --memory 4G \
     --webui-port 8081 --properties-file "%SPARK_HOME%\conf\spark-worker.conf"
   ```
   Dentro de `%SPARK_HOME%\conf\spark-worker.conf` puedes añadir
   `spark.executorEnv.PYSPARK_PYTHON=C:\\strawberry-env\\Scripts\\python.exe`
   para que todos los ejecutores Windows usen ese intérprete. El watcher también puede
   inyectar esta ruta con `--executor-python`.
5. Si quieres que el proceso arranque automáticamente al iniciar sesión, crea un acceso directo a ese comando o programa una tarea en el **Task Scheduler**.
6. Comprueba en `http://<ip_master>:8080` que el worker se muestra activo.

> Para múltiples workers en Windows, duplica el comando ajustando `--webui-port` y preferiblemente ejecutándolo en sesiones separadas.

## 5. Flujo operativo

1. Copia imágenes `.webp` al directorio compartido (`/mnt/strawberry/batch_incoming` en Linux, `S:\\batch_incoming` en Windows).
2. El watcher en el master detecta los archivos, espera la ventana de `--debounce` sin nuevos eventos y lanza `spark-submit` con `src/core/batch.py`.
3. Spark reparte el trabajo entre los workers. Cada worker carga el modelo (`BATCH_MODEL_PATH`) y ejecuta la inferencia sobre su partición.
4. El pipeline crea un Parquet consolidado y un CSV en `/mnt/strawberry/batch_outputs/` y limpia las imágenes ya procesadas.
5. El watcher queda a la espera de la próxima tanda de archivos.

## 6. Monitoreo y validación

- **UI del master** (`http://<ip_master>:8080`): estado del cluster, jobs, workers conectados.
- **UI del worker** (`http://<ip_worker>:8081`): recursos consumidos y tareas asignadas.
- **Logs del watcher**: consola, `journalctl -u strawberry-watcher` (si usas systemd).
- **Resultados**: revisa `batch_outputs/batch_outputs.csv` y `part-00000-*.parquet`.

## 7. Operaciones comunes

| Tarea | Comando |
|-------|---------|
| Detener watcher (systemd) | `sudo systemctl stop strawberry-watcher` |
| Reiniciar master de Spark | `$SPARK_HOME/sbin/stop-master.sh` seguido de `start-master.sh` |
| Escalar worker manualmente | Ejecutar `start-worker.sh` en un nuevo equipo |
| Limpiar directorios | Linux: `rm -rf /mnt/strawberry/batch_outputs/*` · Windows: `Remove-Item S:\\batch_outputs\* -Recurse` |

## 8. Consejos y solución de problemas

- **Archivos incompletos**: si notas que Spark lee imágenes “a medio copiar”, adopta un flujo staging (subir como `.tmp` y renombrar a `.webp` al finalizar) o aumenta `BATCH_WATCH_DEBOUNCE`.
- **Permisos Samba**: usa usuarios dedicados y restringe IPs. En Windows asegúrate de conectar la unidad con credenciales persistentes (opción *Reconnect at sign-in*).
- **Rutas absolutas**: todas las máquinas deben ver las mismas rutas (`/mnt/strawberry/...`). Ajusta variables `BATCH_*` según tus montajes.
- **Jarros Java**: asegúrate de que `JAVA_HOME` apunta al JDK 17 en todos los nodos o Spark no levantará.
- **Workers lentos**: incrementa `SPARK_WORKER_CORES` o agrega más nodos. Puedes quitar `coalesce(1)` dentro del bucle si necesitas mayor paralelismo durante la escritura y mantenerlo solo antes del parquet final.
- **Watcher en contenedor**: si prefieres Docker, crea una imagen ligera con el proyecto y ejecuta `batch_watcher.py` en el master, montando `/mnt/strawberry` dentro del contenedor.

## 9. Check-list final

- [ ] Carpeta compartida montada en todos los nodos (`/mnt/strawberry` en Linux, unidad `S:` en Windows).
- [ ] Master de Spark corriendo y accesible (`spark://<ip_master>:7077`).
- [ ] Workers conectados y visibles en la UI.
- [ ] Watcher activo en el master, apuntando al directorio compartido.
- [ ] Variables `BATCH_*` y `SPARK_*` configuradas en cada máquina.
- [ ] Prueba manual exitosa (copiar imágenes -> parquet + CSV generados).

Con estos pasos tendrás un cluster Spark distribuido con procesamiento batch totalmente automatizado y listo para producción en dos máquinas (o más). Ajusta la memoria, los lotes e intervalos según las necesidades reales de tu flujo de trabajo.
