# --- BASE ---
# Etapa base con dependencias de Python y Sistema
FROM python:3.10-slim AS base
WORKDIR /code
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- BACKEND ---
FROM base AS backend
# Copiar código del backend
COPY src/ ./src/
COPY app/main.py ./app/
# Crear estructura de directorios necesaria para que FastAPI no falle al montar static
RUN mkdir -p app/static/uploads app/static/results
# Exponer puerto
EXPOSE 8000
# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- FRONTEND ---
FROM nginx:alpine AS frontend
# Copiar el HTML principal como index.html
COPY app/pages/app.html /usr/share/nginx/html/index.html
# Copiar los recursos estáticos (JS, CSS, imágenes)
COPY app/static /usr/share/nginx/html/static
# Copiar script de entrada
COPY docker/frontend-entrypoint.sh /docker-entrypoint.sh
# Corregir saltos de línea de Windows (CRLF) a Linux (LF) para evitar errores
RUN sed -i 's/\r$//' /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
# Configuración opcional de Nginx si fuera necesaria (por defecto sirve puerto 80)
EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
