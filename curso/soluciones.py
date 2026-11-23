"""Soluciones de los ejercicios «Tu turno» del curso. Para el profesor.

Cada función resuelve el ejercicio del mismo nombre. Las pruebas automáticas comprueban que
el corrector de `cero.taller` las da por buenas, así que aquí no puede haber una solución mal.

    python curso/soluciones.py      corrige todas las soluciones con los mismos correctores
"""

from __future__ import annotations

import math

import torch


# Clase 1 · El tokenizador -------------------------------------------------------------
def contar_pares(ids):
    cuenta = {}
    for par in zip(ids, ids[1:]):
        cuenta[par] = cuenta.get(par, 0) + 1
    return cuenta


def fusionar(ids, par, nuevo):
    salida, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
            salida.append(nuevo)
            i += 2
        else:
            salida.append(ids[i])
            i += 1
    return salida


# Clase 2 · Datos ----------------------------------------------------------------------
def hacer_lote(tokens, inicios, contexto):
    x = torch.stack([tokens[i : i + contexto] for i in inicios])
    y = torch.stack([tokens[i + 1 : i + 1 + contexto] for i in inicios])
    return x, y


# Clase 3 · Posición -------------------------------------------------------------------
def mi_tabla(largo, d):
    posicion = torch.arange(largo).float().unsqueeze(1)
    divisor = 10000 ** (torch.arange(0, d, 2).float() / d)
    tabla = torch.zeros(largo, d)
    tabla[:, 0::2] = torch.sin(posicion / divisor)
    tabla[:, 1::2] = torch.cos(posicion / divisor)
    return tabla


# Clase 4 · Atención -------------------------------------------------------------------
def mi_mascara(T):
    return torch.ones(T, T).tril().bool()


def mi_atencion(q, k, v, mascara):
    puntos = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    puntos = puntos.masked_fill(~mascara, float("-inf"))
    pesos = torch.softmax(puntos, dim=-1)
    return pesos @ v


# Clase 5 · El bloque ------------------------------------------------------------------
def mi_norma(x, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# Clase 7 · Entrenar -------------------------------------------------------------------
def mi_tasa(paso, lr=1e-3, calentamiento=100, pasos=1000, lr_final=0.1):
    if paso < calentamiento:
        return lr * (paso + 1) / calentamiento
    avance = (paso - calentamiento) / (pasos - calentamiento)
    coseno = 0.5 * (1 + math.cos(math.pi * avance))
    return lr * (lr_final + (1 - lr_final) * coseno)


# Clase 9 · Generar --------------------------------------------------------------------
def con_temperatura(logits, temperatura):
    return torch.softmax(logits / temperatura, dim=-1)


def mi_top_k(logits, k):
    umbral = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < umbral, float("-inf"))


# Clase 10 · Chat ----------------------------------------------------------------------
def mi_plantilla(tok, mensajes):
    from cero.tokenizador import ASISTENTE, FIN, INICIO, USUARIO

    ids, mascara = [tok.id(INICIO)], [0]
    for mensaje in mensajes:
        contenido = tok.codificar_texto(mensaje["texto"])
        if mensaje["rol"] == "usuario":
            ids += [tok.id(USUARIO)] + contenido + [tok.id(FIN)]
            mascara += [0] * (len(contenido) + 2)
        else:
            ids += [tok.id(ASISTENTE)] + contenido + [tok.id(FIN)]
            mascara += [0] + [1] * (len(contenido) + 1)
    return ids, mascara


def corregir_todo() -> bool:
    from cero import taller

    resultados = [
        taller.comprobar_contar_pares(contar_pares), taller.comprobar_fusionar(fusionar), taller.comprobar_lote(hacer_lote),
        taller.comprobar_sinusoidal(mi_tabla), taller.comprobar_mascara(mi_mascara), taller.comprobar_atencion(mi_atencion),
        taller.comprobar_norma_rms(mi_norma), taller.comprobar_tasa(mi_tasa), taller.comprobar_temperatura(con_temperatura),
        taller.comprobar_top_k(mi_top_k), taller.comprobar_plantilla(mi_plantilla),
    ]
    return all(resultados)


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(0 if corregir_todo() else 1)
