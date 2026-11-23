"""El curso: los cuadernos están al día y todas las clases se ejecutan de principio a fin.

Con CERO_RAPIDO=1 las clases encogen sus entrenamientos (pocos pasos, pocos datos), así que
las trece se ejecutan en un par de minutos. Se ejecutan tal como las encontraría un
estudiante: sin haber resuelto los ejercicios «Tu turno».
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
FUENTES = sorted((RAIZ / "curso" / "fuentes").glob("[0-9]*.py"))
sys.path.insert(0, str(RAIZ / "curso"))

import construir  # noqa: E402


def test_hay_trece_clases_numeradas():
    assert [f.name[:2] for f in FUENTES] == [f"{i:02d}" for i in range(13)]


def test_los_cuadernos_estan_al_dia():
    for fuente in FUENTES:
        cuaderno = RAIZ / "curso" / f"{fuente.stem}.ipynb"
        assert cuaderno.exists(), f"Falta {cuaderno.name}: ejecuta  python curso/construir.py"
        esperado = construir.cuaderno(construir.partir(fuente.read_text(encoding="utf-8")))
        assert json.loads(cuaderno.read_text(encoding="utf-8")) == esperado, f"{cuaderno.name} no está al día: python curso/construir.py"


def test_cada_clase_empieza_con_titulo_y_enlaza_la_siguiente():
    for i, fuente in enumerate(FUENTES):
        celdas = construir.partir(fuente.read_text(encoding="utf-8"))
        assert celdas[0][0] == "markdown" and celdas[0][1].startswith(f"# Clase {i} ·"), fuente.name
        if i < len(FUENTES) - 1:
            assert f"({FUENTES[i + 1].stem}.ipynb)" in celdas[-1][1], f"{fuente.name} no enlaza la clase siguiente"


@pytest.mark.lento
@pytest.mark.parametrize("fuente", FUENTES, ids=[f.stem for f in FUENTES])
def test_la_clase_se_ejecuta_entera(fuente, tmp_path_factory):
    carpeta = tmp_path_factory.getbasetemp() / "curso-rapido"          # compartida: las clases reutilizan el proyecto
    entorno = dict(os.environ, CERO_RAPIDO="1", MPLBACKEND="Agg", CERO_DISPOSITIVO="cpu", PYTHONUTF8="1",
                   CERO_PROYECTOS=str(carpeta / "proyectos"), CERO_DESCARGAS=str(carpeta / "descargas"))
    resultado = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", str(fuente)], cwd=RAIZ, env=entorno,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert resultado.returncode == 0, f"{fuente.name} falló:\n{resultado.stdout[-1500:]}\n{resultado.stderr[-3000:]}"
    assert "Traceback" not in resultado.stdout


def test_las_soluciones_pasan_los_correctores(capsys):
    import soluciones

    assert soluciones.corregir_todo(), capsys.readouterr().out


def test_los_correctores_detectan_errores_tipicos(capsys):
    """Un corrector que lo aprueba todo no sirve: con fallos típicos debe suspender y dar una pista."""
    import torch

    from cero import taller

    assert not taller.comprobar_mascara(lambda T: ~torch.ones(T, T).tril().bool())
    assert "al revés" in capsys.readouterr().out
    assert not taller.comprobar_atencion(lambda q, k, v, m: (q @ k.transpose(-2, -1)).masked_fill(~m, float("-inf")).softmax(-1) @ v)
    assert "raíz" in capsys.readouterr().out
    assert not taller.comprobar_norma_rms(lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:]))
    assert "LayerNorm" in capsys.readouterr().out
    assert not taller.comprobar_temperatura(lambda logits, t: (logits * t).softmax(-1))
    assert "DIVIDIR" in capsys.readouterr().out
    assert not taller.comprobar_contar_pares(lambda ids: {})
