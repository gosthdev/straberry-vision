# Strawberry Vision 🍓

Sistema de visión artificial para la detección y clasificación automática de fresas según su estado de madurez. Este proyecto implementa una arquitectura de Deep Learning personalizada (SGSNet) accesible a través de una interfaz web moderna y fácil de usar.

## Características Principales

- **Detección Precisa**: Identifica múltiples fresas en una sola imagen.
- **Clasificación de Madurez**: Categoriza cada detección en 5 estados fenológicos:
  - 🌸 `flowering` (Floración)
  - 🟢 `growing_g` (Crecimiento Verde)
  - 🟡 `growing_w` (Crecimiento Blanco)
  - 🟠 `nearly_m` (Casi Madura)
  - 🔴 `mature` (Madura)
- **Analíticas en Tiempo Real**: Genera gráficos de distribución de madurez y conteo total de frutas.
- **Interfaz Moderna**: Carga de imágenes "Drag & Drop", visualización de resultados con cajas delimitadoras (bounding boxes) y diseño responsivo.
- **Arquitectura Escalable**: Separación clara entre Backend (API de inferencia) y Frontend, optimizada para despliegue con Docker.

## El Modelo de IA

El corazón del sistema es un modelo de red neuronal convolucional (CNN) diseñado para la detección de objetos.

1.  **Entrenamiento**: El modelo se entrena offline utilizando un dataset curado de imágenes de fresas. Durante este proceso, la red aprende a distinguir las características visuales de cada etapa de madurez.
2.  **Validación**: Se monitorean las métricas de precisión y pérdida. El sistema guarda automáticamente el mejor rendimiento en `src/data/processed/models/best_model.pth`.
3.  **Inferencia**: La aplicación web carga este archivo `best_model.pth` para realizar predicciones sobre nuevas imágenes subidas por el usuario.

## Stack Tecnológico

- **Backend**: 
  - Python 3.10
  - **FastAPI**: Para la creación de la API REST.
  - **PyTorch**: Framework de Deep Learning para la inferencia.
  - **OpenCV**: Procesamiento de imágenes.
- **Frontend**: 
  - HTML5 / JavaScript (Vanilla)
  - **TailwindCSS**: Estilizado y diseño responsivo.
  - **Chart.js**: Visualización de datos.
- **DevOps**: 
  - **Docker**: Contenedorización con *multi-stage builds*.
  - **Nginx**: Servidor web para el frontend en producción.

## Instalación y Despliegue

### Opción 1: Despliegue con Docker (Producción)

El proyecto está configurado para desplegarse como dos servicios independientes, cada uno con su propio Dockerfile optimizado:

**1. Backend (API)**
```bash
# Construir la imagen del backend
docker build -f docker/Dockerfile.backend -t strawberry-backend .

# Ejecutar el contenedor (Puerto 8000)
docker run -p 8000:8000 strawberry-backend
```

**2. Frontend (Web)**
```bash
# Construir la imagen del frontend
docker build -f docker/Dockerfile.frontend -t strawberry-frontend .

# Ejecutar conectando al backend (Puerto 80)
# Reemplaza la URL con la dirección real de tu backend desplegado
docker run -p 80:80 -e API_URL="http://localhost:8000" strawberry-frontend
```

### Opción 2: Ejecución Local (Desarrollo)

1.  **Configurar entorno virtual**:
    ```bash
    python -m venv env
    # Windows
    .\env\Scripts\activate
    # Linux/Mac
    source env/bin/activate
    ```

2.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Iniciar el servidor**:
    ```bash
    uvicorn app.main:app --reload
    ```
    La aplicación estará disponible en `http://localhost:8000/static/pages/app.html`.

## 📂 Estructura del Proyecto

```
straberry-vision/
├── app/
│   ├── main.py            # Punto de entrada de la API FastAPI
│   ├── pages/             # Archivos HTML del frontend
│   └── static/            # JavaScript, CSS y assets
├── src/
│   ├── core/              # Definición del modelo (SGSNet), configuración y lógica
│   └── data/              # Rutas de datos y modelos entrenados (best_model.pth)
├── docker/                # Dockerfiles y scripts de configuración
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── frontend-entrypoint.sh
├── requirements.txt       # Dependencias completas (entrenamiento + inferencia)
└── requirements-web.txt   # Dependencias optimizadas para el backend web
```
