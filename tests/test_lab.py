"""El laboratorio: cada dirección de su API responde bien y solo atiende a este mismo equipo."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.lento


@pytest.fixture(scope="module")
def lab(proyecto_mini):
    from cero.lab.servidor import Laboratorio, crear_manejador
    from cero.proyecto import activar

    activar(proyecto_mini.nombre)
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)     # puerto 0 = uno libre cualquiera
    puerto = servidor.server_address[1]
    laboratorio = Laboratorio("cpu")
    servidor.RequestHandlerClass = crear_manejador(laboratorio, puerto)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{puerto}", laboratorio
    servidor.shutdown()
    servidor.server_close()


def pedir(base: str, ruta: str, datos: dict | None = None, cabeceras: dict | None = None, tipo: str = "application/json"):
    cuerpo = None if datos is None else json.dumps(datos).encode("utf-8")
    todas = dict(cabeceras or {})
    if datos is not None:
        todas["Content-Type"] = tipo
    try:
        with urllib.request.urlopen(urllib.request.Request(base + ruta, data=cuerpo, headers=todas), timeout=120) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def json_de(base: str, ruta: str, datos: dict | None = None) -> dict:
    codigo, texto = pedir(base, ruta, datos)
    assert codigo == 200, texto
    return json.loads(texto)


def eventos_de(base: str, ruta: str, datos: dict) -> list[dict]:
    codigo, texto = pedir(base, ruta, datos)
    assert codigo == 200, texto
    return [json.loads(linea) for linea in texto.splitlines() if linea.strip()]


def test_la_pagina_y_sus_archivos(lab):
    base, _ = lab
    codigo, html = pedir(base, "/")
    assert codigo == 200 and "Laboratorio" in html
    assert pedir(base, "/app.js")[0] == 200 and pedir(base, "/estilo.css")[0] == 200
    assert pedir(base, "/no-existe.js")[0] == 404
    assert pedir(base, "/..%2f..%2fconfig.py")[0] == 404       # nada fuera de la carpeta de la página
    assert pedir(base, "/api/desconocida")[0] == 404


def test_estado(lab, proyecto_mini):
    base, _ = lab
    estado = json_de(base, "/api/estado")
    assert estado["activo"] == proyecto_mini.nombre
    assert estado["proyecto"]["base_listo"] and estado["proyecto"]["chat_listo"]
    assert set(estado["recetas"]) == {"nano", "micro", "mini", "base"}
    assert estado["tarea"] == {"activa": False}


def test_tokenizador_y_modelo(lab):
    base, _ = lab
    texto = "Había una vez una gata. ¿Cuánto es 7 + 8?"
    r = json_de(base, "/api/tokenizar", {"texto": texto})
    assert r["tokens"] == len(r["piezas"]) and r["caracteres"] == len(texto)
    assert b"".join(bytes(p["bytes"]) for p in r["piezas"]).decode("utf-8") == texto
    fusiones = json_de(base, "/api/fusiones?desde=0&cuantas=10")
    assert len(fusiones["fusiones"]) == 10 and fusiones["fusiones"][0]["id"] == 256
    assert {e["texto"] for e in fusiones["especiales"]} == {"<|inicio|>", "<|usuario|>", "<|asistente|>", "<|fin|>"}
    modelo = json_de(base, "/api/modelo")
    assert sum(v for k, v in modelo["reparto"].items() if k != "total") == modelo["reparto"]["total"]
    assert json_de(base, "/api/datos")["datos"]["partes"]["entreno"]["tokens"] > 0


def test_atencion(lab):
    base, _ = lab
    r = json_de(base, "/api/atencion", {"texto": "Había una vez un gato pequeño."})
    n = len(r["tokens"])
    assert r["tokens"][0] == "<|inicio|>" and len(r["atencion"]) == 2 and len(r["atencion"][0]) == 4
    matriz = r["atencion"][1][2]
    assert len(matriz) == n and all(len(fila) == n for fila in matriz)
    assert all(abs(sum(fila) - 1000) <= n for fila in matriz)               # cada fila reparte un 100 % (en tantos por mil)
    assert all(matriz[i][j] == 0 for i in range(n) for j in range(i + 1, n))   # nada hacia el futuro


def test_candidatos_y_generacion(lab):
    base, _ = lab
    r = json_de(base, "/api/candidatos", {"texto": "Había una", "temperatura": 0.7, "top_k": 5})
    probs = [c["prob"] for c in r["candidatos"]]
    assert probs == sorted(probs, reverse=True) and sum(c["prob_ajustada"] > 0 for c in r["candidatos"]) == 5
    eventos = eventos_de(base, "/api/generar", {"texto": "Había una vez", "max": 12, "semilla": 3, "temperatura": 0.8})
    assert eventos[0]["etapa"] in ("base", "chat") and eventos[-1]["fin"] and eventos[-1]["tokens"] <= 12
    otra_vez = eventos_de(base, "/api/generar", {"texto": "Había una vez", "max": 12, "semilla": 3, "temperatura": 0.8})
    assert [e.get("trozo") for e in eventos] == [e.get("trozo") for e in otra_vez]     # misma semilla, mismo texto


def test_chat(lab):
    base, _ = lab
    mensajes = [{"rol": "usuario", "texto": "Hola"}]
    eventos = eventos_de(base, "/api/chat", {"mensajes": mensajes, "max": 10, "temperatura": 0.5, "semilla": 1})
    visto = [t["texto"] for t in eventos[0]["visto"]]
    assert visto[:2] == ["<|inicio|>", "<|usuario|>"] and visto[-1] == "<|asistente|>"
    assert eventos[0]["tokens_entrada"] == len(visto) and eventos[-1]["fin"]
    plantilla = json_de(base, "/api/plantilla", {"mensajes": mensajes + [{"rol": "asistente", "texto": "¡Hola!"}]})["tokens"]
    assert [t["texto"] for t in plantilla if t["aprende"]][-1] == "<|fin|>"
    assert not any(t["aprende"] for t in plantilla[:4])


def test_metricas(lab):
    base, _ = lab
    r = json_de(base, "/api/metricas?etapa=base&desde=0")
    assert r["total"] == len(r["eventos"]) > 0 and r["eventos"][0]["tipo"] == "eval" and r["pasos"] == 24
    assert json_de(base, f"/api/metricas?etapa=base&desde={r['total']}")["eventos"] == []


def test_solo_atiende_a_este_equipo(lab):
    base, _ = lab
    assert pedir(base, "/api/estado", cabeceras={"Host": "malo.example"})[0] == 403          # otra máquina o DNS trucado
    assert pedir(base, "/api/parar", {}, cabeceras={"Origin": "http://malo.example"})[0] == 403   # otra página web
    assert pedir(base, "/api/parar", {}, tipo="text/plain")[0] == 415                        # formulario de otra web
    assert pedir(base, "/api/usar")[0] == 404                                                # las órdenes no van por GET


def test_errores_de_la_api(lab):
    base, _ = lab
    codigo, texto = pedir(base, "/api/parar", {})
    assert codigo == 400 and "No hay ningún entrenamiento" in texto
    assert pedir(base, "/api/lanzar", {"etapa": "inventada"})[0] == 400
    assert pedir(base, "/api/usar", {"nombre": "no-existe"})[0] == 400
    assert pedir(base, "/api/tokenizar", {"texto": "x" * 5000})[0] == 400
    assert pedir(base, "/api/chat", {"mensajes": [{"rol": "sistema", "texto": "x"}]})[0] == 400


def test_lanzar_una_tarea(lab):
    _, laboratorio = lab
    estado = laboratorio.lanzar(["recetas"], "prueba", None)
    assert estado["activa"] or estado["codigo"] == 0
    with pytest.raises(ValueError):
        if laboratorio.tarea_activa():
            laboratorio.lanzar(["recetas"], "otra", None)       # solo una tarea a la vez
        else:
            raise ValueError
    for _ in range(600):
        if not laboratorio.tarea_activa():
            break
        time.sleep(0.1)
    final = laboratorio.estado_tarea()
    assert final["codigo"] == 0 and "Recetas disponibles" in final["registro"]
