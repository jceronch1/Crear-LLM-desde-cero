"""Preparación común de las pruebas.

Las pruebas nunca tocan la carpeta `proyectos/` real: trabajan en una carpeta temporal.
`proyecto_mini` es un proyecto diminuto pero completo (datos, modelo base y chatbot),
entrenado unos segundos en CPU; lo comparten todas las pruebas marcadas como `lento`.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def carpeta_de_pruebas(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("cero")
    os.environ["CERO_PROYECTOS"] = str(raiz / "proyectos")
    os.environ["CERO_DESCARGAS"] = str(raiz / "descargas")
    torch.set_num_threads(2)                 # resultados iguales en cualquier equipo y sin acaparar la CPU
    return raiz


def receta_mini() -> dict:
    """La receta nano encogida para que todo el recorrido dure pocos segundos."""
    from cero.config import receta

    r = receta("nano")
    r["corpus"]["cuentos"] = 1500
    r["vocab"] = 420
    r["modelo"].update(vocab=420, contexto=192, d_modelo=48, n_capas=2, n_cabezas=4)
    r["entreno"].update(pasos=24, lote=8, evaluar_cada=12, lotes_eval=2, calentamiento=4)
    r["chat"].update(pasos=12, lote=8, evaluar_cada=6, lotes_eval=2, calentamiento=2)
    r["dialogos"] = 500
    return r


@pytest.fixture(scope="session")
def proyecto_mini(carpeta_de_pruebas):
    from cero import chat, datos, entrenar
    from cero.proyecto import Proyecto

    proyecto = Proyecto.crear("mini-prueba", receta_mini(), reemplazar=True)
    silencio = lambda _mensaje: None  # noqa: E731
    datos.preparar(proyecto, silencio)
    entrenar.entrenar_base(proyecto, "cpu", avisar=silencio)
    chat.afinar(proyecto, "cpu", avisar=silencio)
    return proyecto


@pytest.fixture(scope="session")
def tokenizador():
    from cero.cuentos import generar_cuentos
    from cero.tokenizador import TokenizadorBPE

    return TokenizadorBPE.entrenar(list(generar_cuentos(300, 1)), 400)
