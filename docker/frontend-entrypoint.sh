#!/bin/sh

# Generar config.js basado en variables de entorno
echo "window.APP_CONFIG = { API_URL: '${API_URL:-}' };" > /usr/share/nginx/html/static/js/config.js

# Ejecutar el comando original (nginx)
exec "$@"