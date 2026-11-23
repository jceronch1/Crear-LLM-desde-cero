"""Convierte las fuentes del curso (curso/fuentes/*.py) en cuadernos de Jupyter (curso/*.ipynb).

Las clases se escriben como archivos de Python normales, con marcas de celda:

    # %% [markdown]        empieza una celda de texto (cada línea va precedida de «# »)
    # %%                   empieza una celda de código

Así se pueden revisar, comparar y ejecutar como cualquier programa (las pruebas automáticas
los ejecutan todos), y de ellos salen los cuadernos que abre el estudiante.

    python curso/construir.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MARCA = re.compile(r"^# %%(.*)$")


def partir(texto: str) -> list[tuple[str, str]]:
    """Devuelve la lista de celdas como (tipo, contenido); tipo es «markdown» o «code»."""
    celdas: list[tuple[str, list[str]]] = []
    for linea in texto.splitlines():
        marca = MARCA.match(linea)
        if marca:
            celdas.append(("markdown" if "[markdown]" in marca.group(1) else "code", []))
        elif celdas:                                         # lo anterior a la primera marca se descarta
            celdas[-1][1].append(linea)
    resultado = []
    for tipo, lineas in celdas:
        if tipo == "markdown":
            lineas = [linea[2:] if linea.startswith("# ") else linea[1:] if linea.startswith("#") else linea for linea in lineas]
        contenido = "\n".join(lineas).strip("\n")
        if contenido.strip():
            resultado.append((tipo, contenido))
    return resultado


def cuaderno(celdas: list[tuple[str, str]]) -> dict:
    lista = []
    for i, (tipo, contenido) in enumerate(celdas):
        celda = {"cell_type": tipo, "id": f"celda-{i:03d}", "metadata": {}, "source": contenido.splitlines(keepends=True)}
        if tipo == "code":
            celda.update(execution_count=None, outputs=[])
        lista.append(celda)
    return {
        "cells": lista,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def construir() -> list[Path]:
    creados = []
    for fuente in sorted((AQUI / "fuentes").glob("[0-9]*.py")):
        destino = AQUI / f"{fuente.stem}.ipynb"
        contenido = json.dumps(cuaderno(partir(fuente.read_text(encoding="utf-8"))), ensure_ascii=False, indent=1) + "\n"
        if not destino.exists() or destino.read_text(encoding="utf-8") != contenido:
            destino.write_text(contenido, encoding="utf-8")
        creados.append(destino)
    return creados


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    for ruta in construir():
        print("  ", ruta.name)
