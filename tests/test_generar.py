"""La generación: los mandos de muestreo hacen lo que dicen y la caché no cambia el texto."""

from __future__ import annotations

import pytest
import torch

from cero.config import ConfigModelo
from cero.generar import candidatos, elegir, filtrar, generar
from cero.modelo import GPT


@pytest.fixture(scope="module")
def modelo() -> GPT:
    torch.manual_seed(0)
    return GPT(ConfigModelo(vocab=300, contexto=32, d_modelo=64, n_capas=2, n_cabezas=4)).eval()


LOGITS = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])


def test_top_k_deja_exactamente_k():
    probs = filtrar(LOGITS, top_k=2).softmax(-1)
    assert (probs > 0).sum() == 2 and probs[2:].sum() == 0
    assert probs.sum().item() == pytest.approx(1.0)


def test_top_p_deja_el_nucleo_mas_pequeno():
    originales = LOGITS.softmax(-1)                         # 0.56, 0.21, 0.12, 0.08, 0.03
    assert (filtrar(LOGITS, top_p=0.5).softmax(-1) > 0).sum() == 1      # el primero ya suma más de 0.5
    assert (filtrar(LOGITS, top_p=0.6).softmax(-1) > 0).sum() == 2
    assert (filtrar(LOGITS, top_p=0.8).softmax(-1) > 0).sum() == 3
    assert originales[:3].sum() > 0.8 > originales[:2].sum()


def test_la_temperatura_afila_o_aplana():
    fria, normal, caliente = (filtrar(LOGITS, t).softmax(-1)[0].item() for t in (0.5, 1.0, 2.0))
    assert fria > normal > caliente


def test_temperatura_cero_es_el_mas_probable():
    assert elegir(LOGITS, temperatura=0) == 0
    assert elegir(torch.tensor([0.1, 0.2, 5.0, 0.3]), temperatura=0) == 2


def test_misma_semilla_mismo_texto_con_y_sin_cache(modelo):
    inicio = [5, 17, 200]
    con_cache = list(generar(modelo, inicio, 30, semilla=3, usar_cache=True))
    sin_cache = list(generar(modelo, inicio, 30, semilla=3, usar_cache=False))
    assert con_cache == sin_cache and len(con_cache) == 30
    assert list(generar(modelo, inicio, 30, semilla=4)) != con_cache


def test_puede_escribir_mas_que_el_contexto(modelo):
    assert len(list(generar(modelo, [1, 2, 3], 100, semilla=0))) == 100


def test_se_detiene_en_el_token_de_parada(modelo):
    libre = list(generar(modelo, [1, 2, 3], 40, temperatura=0))
    parada = libre[5]
    cortado = list(generar(modelo, [1, 2, 3], 40, temperatura=0, parar=[parada]))
    assert cortado == libre[: libre.index(parada)]
    assert parada not in cortado


def test_no_deja_el_modelo_en_otro_modo(modelo):
    modelo.train()
    list(generar(modelo, [1], 3))
    assert modelo.training
    modelo.eval()


def test_necesita_al_menos_un_token(modelo):
    with pytest.raises(ValueError):
        list(generar(modelo, [], 3))


def test_candidatos(modelo, tokenizador):
    lista = candidatos(modelo, tokenizador, [1, 2, 3], temperatura=0.7, top_k=5, cuantos=10)
    assert len(lista) == 10
    assert [c["prob"] for c in lista] == sorted((c["prob"] for c in lista), reverse=True)
    assert sum(c["prob_ajustada"] > 0 for c in lista) == 5           # top-k = 5 deja cinco con opciones
    assert sum(c["prob_ajustada"] for c in lista) == pytest.approx(1.0, abs=1e-4)
