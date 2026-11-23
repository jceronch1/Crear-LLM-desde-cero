#!/usr/bin/env sh
# Lanzador para Linux y macOS:  sh cero.sh <orden>     (por ejemplo:  sh cero.sh todo --receta nano)
RAIZ="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$RAIZ/.venv/bin/python" ]; then
    echo "  Falta instalar. Ejecuta primero:  sh instalar.sh"
    exit 1
fi
exec "$RAIZ/.venv/bin/python" -X utf8 -m cero "$@"
