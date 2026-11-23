"""El entrenamiento: la tasa de aprendizaje, que de verdad aprende, y pausar/reanudar."""

from __future__ import annotations

import json
import math

import pytest
import torch

from cero import datos, entrenar
from cero.config import ConfigEntreno
from cero.proyecto import Proyecto, cargar_modelo, leer_punto


def test_la_tasa_sube_y_luego_baja_en_coseno():
    cfg = ConfigEntreno(pasos=100, lr=1e-3, calentamiento=10, lr_final=0.1)
    tasas = [entrenar.tasa_de_aprendizaje(p, cfg) for p in range(100)]
    assert tasas[0] == pytest.approx(1e-4) and tasas[9] == pytest.approx(1e-3)
    assert all(b > a for a, b in zip(tasas[:10], tasas[1:10]))          # sube durante el calentamiento
    assert all(b < a for a, b in zip(tasas[10:], tasas[11:]))           # y después solo baja
    assert max(tasas) == pytest.approx(1e-3)
    assert entrenar.tasa_de_aprendizaje(100, cfg) == pytest.approx(1e-4)
    assert tasas[55] == pytest.approx(1e-3 * (0.1 + 0.9 * 0.5), rel=1e-6)    # a mitad del coseno


def test_el_decaimiento_solo_toca_las_matrices():
    from cero.config import ConfigModelo
    from cero.modelo import GPT

    modelo = GPT(ConfigModelo(vocab=300, contexto=16, d_modelo=32, n_capas=1, n_cabezas=2))
    optimizador = entrenar.crear_optimizador(modelo, ConfigEntreno(decaimiento=0.1), "cpu")
    con, sin = optimizador.param_groups
    assert con["weight_decay"] == 0.1 and all(p.dim() >= 2 for p in con["params"])
    assert sin["weight_decay"] == 0.0 and all(p.dim() < 2 for p in sin["params"])
    assert sum(p.numel() for g in optimizador.param_groups for p in g["params"]) == modelo.n_parametros()


@pytest.mark.lento
def test_el_modelo_aprende(proyecto_mini):
    eventos = entrenar.leer_metricas(proyecto_mini.etapa("base"))
    evaluaciones = [e for e in eventos if e["tipo"] == "eval"]
    vocab = proyecto_mini.config["modelo"]["vocab"]
    assert evaluaciones[0]["paso"] == 0
    assert evaluaciones[0]["perdida_val"] == pytest.approx(math.log(vocab), abs=0.4)     # sin entrenar: azar
    assert evaluaciones[-1]["perdida_val"] < evaluaciones[0]["perdida_val"] - 1.0        # entrenado: mucho mejor
    assert isinstance(evaluaciones[-1]["muestra"], str)
    estado = json.loads((proyecto_mini.etapa("base") / "estado.json").read_text(encoding="utf-8"))
    assert estado["estado"] == "terminado" and estado["paso"] == estado["pasos"] == 24
    # El ajuste para chat parte del modelo base y deja su propio modelo.
    assert proyecto_mini.estado()["chat_listo"]


@pytest.mark.lento
def test_el_punto_de_control_es_autosuficiente_y_seguro(proyecto_mini):
    ruta = proyecto_mini.mejor("base")
    punto = leer_punto(ruta)                                 # weights_only=True: solo tensores y datos simples
    assert {"modelo", "config_modelo", "tokenizador", "paso", "perdida_val"} <= set(punto)
    modelo, tok, ficha = cargar_modelo(ruta)
    assert tok.vocab == modelo.cfg.vocab and ficha["etapa"] == "base"
    assert not modelo.training


@pytest.mark.lento
def test_pausar_y_reanudar_da_el_mismo_modelo(proyecto_mini):
    """Entrenar 16 pasos seguidos = entrenar 8, pausar y reanudar otros 8."""
    config = proyecto_mini.config
    config["entreno"].update(pasos=16, evaluar_cada=8)
    resultados = {}
    for nombre in ("seguido", "en-dos-partes"):
        proyecto = Proyecto.crear(f"reanudar-{nombre}", config, reemplazar=True)
        datos.preparar(proyecto, lambda _m: None)
        if nombre == "seguido":
            entrenar.entrenar_base(proyecto, "cpu", avisar=lambda _m: None)
        else:
            pasos_hechos = iter(range(1, 100))
            pausado = entrenar.entrenar_base(proyecto, "cpu", avisar=lambda _m: None, debe_parar=lambda: next(pasos_hechos) == 8)
            assert pausado["estado"] == "pausado" and pausado["paso"] == 8
            final = entrenar.entrenar_base(proyecto, "cpu", reanudar=True, avisar=lambda _m: None)
            assert final["estado"] == "terminado" and final["paso"] == 16
        resultados[nombre] = leer_punto(proyecto.etapa("base") / "ultimo.pt")["modelo"]
    for clave, pesos in resultados["seguido"].items():
        assert torch.allclose(pesos, resultados["en-dos-partes"][clave], atol=1e-5), clave


@pytest.mark.lento
def test_medir_velocidad():
    from cero.config import ConfigModelo

    medida = entrenar.medir_velocidad(ConfigModelo(vocab=300, contexto=32, d_modelo=32, n_capas=1, n_cabezas=2), 4, "cpu", segundos=0.3)
    assert medida["tokens_s"] > 0 and medida["pasos_s"] > 0
