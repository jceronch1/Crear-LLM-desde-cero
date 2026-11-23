"""Del modelo al texto: cómo se elige cada token que se escribe.

El modelo no escribe: solo da, para cada token del vocabulario, una puntuación (*logit*) de
lo bien que encaja como siguiente. Generar es repetir tres pasos:

    1. pasar el texto por el modelo y quedarse con los logits de la ÚLTIMA posición
    2. convertirlos en probabilidades y sortear un token
    3. añadirlo al texto y volver al paso 1

Los mandos del paso 2 actúan en este orden:

    temperatura  divide los logits. < 1 afila la distribución (texto más previsible);
                 > 1 la aplana (más sorpresa y más errores). 0 = elegir siempre el más probable.
    top_k        conserva solo los k tokens más probables.
    top_p        conserva el grupo más pequeño de tokens que juntos suman probabilidad p
                 («muestreo de núcleo»): pocos cuando el modelo está seguro, muchos cuando duda.

El artículo de 2017 usaba otra estrategia (búsqueda en haz, §6.1), adecuada para traducir.
Para escribir texto abierto hoy se muestrea, que es lo que hace este archivo.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor

from .modelo import GPT
from .tokenizador import TokenizadorBPE


def filtrar(logits: Tensor, temperatura: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> Tensor:
    """Aplica temperatura, top-k y top-p. Los tokens descartados quedan a -infinito (probabilidad 0)."""
    logits = logits.float() / max(float(temperatura), 1e-6)
    if top_k is not None and 0 < top_k < logits.size(-1):
        umbral = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]    # el k-ésimo logit más alto
        logits = logits.masked_fill(logits < umbral, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        ordenados, orden = torch.sort(logits, descending=True, dim=-1)
        probs = ordenados.softmax(dim=-1)
        acumulado_antes = probs.cumsum(dim=-1) - probs       # probabilidad sumada por los tokens anteriores
        fuera = acumulado_antes >= top_p                     # el núcleo ya estaba completo sin este token
        fuera = fuera.scatter(-1, orden, fuera)              # de vuelta al orden original del vocabulario
        logits = logits.masked_fill(fuera, float("-inf"))
    return logits


def elegir(
    logits: Tensor,
    temperatura: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generador: torch.Generator | None = None,
) -> int:
    """Elige el siguiente token a partir de los logits (vocab,) de la última posición."""
    logits = logits.reshape(-1)
    if temperatura <= 0:
        return int(logits.argmax())                          # «voraz»: siempre el más probable
    probs = filtrar(logits, temperatura, top_k, top_p).softmax(dim=-1)
    # El sorteo se hace en la CPU: así una misma semilla da lo mismo en cualquier equipo.
    return int(torch.multinomial(probs.cpu(), num_samples=1, generator=generador))


@torch.inference_mode()
def generar(
    modelo: GPT,
    ids: Iterable[int],
    max_nuevos: int = 100,
    temperatura: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    parar: Iterable[int] = (),
    semilla: int | None = None,
    usar_cache: bool = True,
) -> Iterator[int]:
    """Va entregando tokens nuevos, uno a uno, a continuación de `ids`.

    Se detiene al llegar a `max_nuevos` o cuando sale un token de `parar` (que no se entrega).
    Es un generador de Python: `for token in generar(...)` permite mostrar el texto mientras se escribe.

    Con `usar_cache=True` (lo normal) el texto inicial se procesa una vez y cada paso posterior
    solo calcula el token nuevo. Si la ventana de contexto se llena, se conserva la mitad más
    reciente y se sigue: el modelo «olvida» lo más antiguo.
    """
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError("Hace falta al menos un token para empezar a generar (por ejemplo <|inicio|>).")
    parar = {int(p) for p in parar}
    contexto = modelo.cfg.contexto
    device = next(modelo.parameters()).device
    generador = None
    if semilla is not None:
        generador = torch.Generator().manual_seed(int(semilla))
    estaba_entrenando = modelo.training
    modelo.eval()                                            # apaga el dropout mientras se genera

    def tensor(tokens: list[int]) -> Tensor:
        return torch.tensor([tokens], dtype=torch.long, device=device)

    try:
        cache = modelo.nueva_cache() if usar_cache else None
        ultimo = modelo.estados(tensor(ids[-contexto:]), cache)[:, -1]       # «prellenado» con el texto inicial
        for _ in range(int(max_nuevos)):
            token = elegir(modelo.salida(ultimo), temperatura, top_k, top_p, generador)
            if token in parar:
                break
            yield token
            ids.append(token)
            if cache is None:
                ultimo = modelo.estados(tensor(ids[-contexto:]))[:, -1]      # sin caché: todo otra vez
            elif cache.longitud >= contexto:
                cache = modelo.nueva_cache()                                 # ventana llena: empezar con la mitad reciente
                ultimo = modelo.estados(tensor(ids[-(contexto // 2):]), cache)[:, -1]
            else:
                ultimo = modelo.estados(tensor([token]), cache)[:, -1]       # con caché: solo el token nuevo
    finally:
        modelo.train(estaba_entrenando)


def completar(
    modelo: GPT,
    tok: TokenizadorBPE,
    texto: str = "",
    max_nuevos: int = 100,
    temperatura: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    semilla: int | None = None,
    sin_parar: bool = False,
) -> str:
    """Continúa un texto y devuelve solo la parte nueva. Para al terminar el documento.

    Con `sin_parar=True` no se detiene en los tokens especiales (útil para ver lo que escribe
    un modelo sin entrenar, que los suelta en cualquier sitio).
    """
    ids = [tok.id("<|inicio|>")] + tok.codificar_texto(texto)
    parar = () if sin_parar else tok.especiales.values()
    nuevos = generar(modelo, ids, max_nuevos, temperatura, top_k, top_p, parar=parar, semilla=semilla)
    return tok.decodificar(list(nuevos))


@torch.inference_mode()
def candidatos(
    modelo: GPT,
    tok: TokenizadorBPE,
    ids: list[int],
    temperatura: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    cuantos: int = 12,
) -> list[dict[str, Any]]:
    """Los tokens más probables para continuar `ids`, antes y después de aplicar los mandos.

    Sirve para VER el paso 2: qué opciones baraja el modelo y cómo las recortan la
    temperatura, top-k y top-p. Lo usan la clase 9 y el laboratorio.
    """
    device = next(modelo.parameters()).device
    estaba_entrenando = modelo.training
    modelo.eval()
    try:
        x = torch.tensor([ids[-modelo.cfg.contexto:]], dtype=torch.long, device=device)
        logits = modelo.salida(modelo.estados(x)[:, -1]).reshape(-1).float()
    finally:
        modelo.train(estaba_entrenando)
    originales = logits.softmax(dim=-1)
    ajustadas = filtrar(logits, temperatura if temperatura > 0 else 1e-6, top_k, top_p).softmax(dim=-1)
    mejores = torch.topk(originales, min(cuantos, originales.numel()))
    return [
        {
            "id": int(i),
            "texto": tok.texto_de(int(i)),
            "logit": float(logits[i]),
            "prob": float(p),                    # probabilidad «tal cual» (temperatura 1, sin recortes)
            "prob_ajustada": float(ajustadas[i]),  # la que de verdad se usa en el sorteo
        }
        for p, i in zip(mejores.values, mejores.indices)
    ]
