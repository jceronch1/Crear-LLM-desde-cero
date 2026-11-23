#!/usr/bin/env sh
# Instalador para Linux y macOS:  sh instalar.sh   (opciones: --cpu  --gpu  --sin-curso)
# Busca un Python 3.11 o posterior (prefiere 3.12, la versión más probada) y ejecuta instalar.py.
cd "$(dirname "$0")" || exit 1
for candidato in python3.12 python3.13 python3.11 python3.14 python3 python; do
    if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        exec "$candidato" instalar.py "$@"
    fi
done
echo "  No se encuentra Python 3.11 o posterior."
echo "  Instálalo (por ejemplo: sudo apt install python3.12 python3.12-venv, o desde python.org) y vuelve a ejecutar:  sh instalar.sh"
exit 1
