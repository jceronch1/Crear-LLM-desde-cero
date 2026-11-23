"""El modelo: las propiedades que tienen que cumplirse siempre, para las tres arquitecturas."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cero.config import ARQUITECTURAS, ConfigModelo, ErrorDeConfig
from cero.modelo import GPT, NormaRMS, atencion, mascara_causal, rotar, tabla_rope, tabla_sinusoidal


def crear(arquitectura: str, **cambios) -> GPT:
    torch.manual_seed(0)
    ajustes = dict(vocab=300, contexto=32, d_modelo=64, n_capas=2, n_cabezas=4) | ARQUITECTURAS[arquitectura] | cambios
    return GPT(ConfigModelo(**ajustes)).eval()


@pytest.fixture(params=list(ARQUITECTURAS))
def modelo(request) -> GPT:
    return crear(request.param)


def tokens(lote: int = 2, largo: int = 32) -> torch.Tensor:
    return torch.randint(0, 300, (lote, largo), generator=torch.Generator().manual_seed(1))


# ---------------------------------------------------------------------------- piezas sueltas
def test_atencion_a_mano_coincide_con_pytorch():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 4, 10, 16).unbind(0)
    a_mano, pesos = atencion(q, k, v, mascara_causal(10, 10))
    de_pytorch = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(a_mano, de_pytorch, atol=1e-5)
    assert torch.allclose(pesos.sum(-1), torch.ones(2, 4, 10), atol=1e-5)        # cada fila reparte un 100 %
    assert float(pesos.triu(1).abs().max()) == 0.0                              # nadie mira al futuro


def test_mascara_causal():
    assert mascara_causal(3, 3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    # Con caché: 2 consultas nuevas frente a 5 claves (3 guardadas + las 2 nuevas).
    assert mascara_causal(2, 5).tolist() == [[True, True, True, True, False], [True, True, True, True, True]]


def test_tabla_sinusoidal_sigue_la_formula_del_articulo():
    tabla = tabla_sinusoidal(50, 16)
    for pos, i in [(0, 0), (7, 2), (33, 5)]:
        angulo = pos / 10000 ** (2 * i / 16)
        assert tabla[pos, 2 * i].item() == pytest.approx(math.sin(angulo), abs=1e-5)
        assert tabla[pos, 2 * i + 1].item() == pytest.approx(math.cos(angulo), abs=1e-5)


def test_rope_solo_depende_de_la_distancia():
    cos, sin = tabla_rope(64, 16)
    q, k = torch.randn(16), torch.randn(16)

    def producto(m: int, n: int) -> float:
        return float((rotar(q, cos[m], sin[m]) * rotar(k, cos[n], sin[n])).sum())

    assert producto(5, 2) == pytest.approx(producto(25, 22), abs=1e-4) == pytest.approx(producto(60, 57), abs=1e-4)
    assert abs(producto(5, 2) - producto(5, 3)) > 1e-3
    assert torch.allclose(rotar(q, cos[9], sin[9]).norm(), q.norm(), atol=1e-5)  # girar no cambia el tamaño


def test_norma_rms():
    x = torch.randn(4, 10, 32) * 7
    y = NormaRMS(32)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 10), atol=1e-3)


# ---------------------------------------------------------------------------- el modelo entero
def test_formas_y_perdida_inicial(modelo):
    x = tokens()
    y = torch.randint(0, 300, (2, 32), generator=torch.Generator().manual_seed(2))
    logits, perdida = modelo(x, y)
    assert logits.shape == (2, 32, 300)
    # Un modelo sin entrenar reparte la probabilidad casi por igual: pérdida ≈ ln(vocabulario).
    assert perdida.item() == pytest.approx(math.log(300), abs=0.35)


def test_el_camino_rapido_y_el_didactico_dan_lo_mismo(modelo):
    x = tokens()
    rapido, _ = modelo(x)
    a_mano, atenciones = modelo.mirar(x)
    assert torch.allclose(rapido, a_mano, atol=1e-4)
    assert len(atenciones) == 2 and atenciones[0].shape == (2, 4, 32, 32)


def test_es_causal(modelo):
    x = tokens()
    otro = x.clone()
    otro[:, 20:] = (otro[:, 20:] + 1) % 300                 # cambiar el futuro...
    assert torch.allclose(modelo(x)[0][:, :20], modelo(otro)[0][:, :20], atol=1e-5)   # ...no cambia el pasado


def test_la_cache_kv_no_cambia_el_resultado(modelo):
    x = tokens(1, 24)
    completo = modelo.estados(x)
    cache = modelo.nueva_cache()
    por_partes = torch.cat([modelo.estados(x[:, :9], cache), modelo.estados(x[:, 9:10], cache),
                            modelo.estados(x[:, 10:], cache)], dim=1)
    assert cache.longitud == 24
    assert torch.allclose(completo, por_partes, atol=1e-4)


def test_no_cabe_mas_que_el_contexto(modelo):
    with pytest.raises(ValueError, match="solo ve"):
        modelo(tokens(1, 33))


def test_los_objetivos_a_menos_uno_no_cuentan(modelo):
    x = tokens()
    y = x.clone()
    y[:, :16] = -1
    _, mitad = modelo(x, y)
    logits, _ = modelo(x)
    esperada = F.cross_entropy(logits[:, 16:].reshape(-1, 300), x[:, 16:].reshape(-1))
    assert mitad.item() == pytest.approx(esperada.item(), abs=1e-4)


def test_pesos_atados_y_recuento():
    atado, suelto = crear("moderna"), crear("moderna", atar_pesos=False)
    assert atado.salida.weight is atado.embeddings.weight
    assert suelto.n_parametros() - atado.n_parametros() == 300 * 64
    reparto = atado.reparto()
    assert sum(v for k, v in reparto.items() if k != "total") == reparto["total"] == atado.n_parametros()
    # A mano, para la arquitectura moderna: embeddings + por bloque (4 matrices d×d, 3 matrices de la red, 2 ganancias) + ganancia final.
    d, oculta = 64, atado.cfg.d_oculta
    assert reparto["total"] == 300 * d + 2 * (4 * d * d + 3 * d * oculta + 2 * d) + d


def test_aprende_un_lote_de_memoria():
    """La comprobación clásica: si el modelo y el gradiente están bien, un lote fijo se memoriza."""
    modelo = crear("moderna").train()
    x = tokens(4, 32)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    primera = None
    for _ in range(80):
        _, perdida = modelo(x[:, :-1], x[:, 1:])
        primera = primera or perdida.item()
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
    assert perdida.item() < 0.25 * primera


def test_configuraciones_imposibles():
    with pytest.raises(ErrorDeConfig, match="divisible"):
        ConfigModelo(d_modelo=100, n_cabezas=3)
    with pytest.raises(ErrorDeConfig, match="par"):
        ConfigModelo(d_modelo=60, n_cabezas=4)
    with pytest.raises(ErrorDeConfig, match="posicion"):
        ConfigModelo(posicion="magica")
    with pytest.raises(ErrorDeConfig, match="desconocidos"):
        ConfigModelo.desde_dict({"capas": 3})
