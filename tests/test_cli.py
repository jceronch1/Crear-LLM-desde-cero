"""La línea de órdenes: cada orden funciona y los errores se explican sin volcados de código."""

from __future__ import annotations

import pytest

from cero.cli import main
from cero.config import ORDEN
from cero.proyecto import Proyecto, activar, listar


def test_sin_argumentos_muestra_la_ayuda(capsys):
    assert main([]) == 0
    assert f"{ORDEN} todo" in capsys.readouterr().out


def test_recetas(capsys):
    assert main(["recetas"]) == 0
    salida = capsys.readouterr().out
    assert all(nombre in salida for nombre in ("nano", "micro", "mini", "base", "paper2017"))


def test_errores_explicados(capsys):
    assert main(["estado", "-p", "no-existe"]) == 1
    assert "No existe el proyecto" in capsys.readouterr().err
    assert main(["nuevo", "nombre con espacios"]) == 1
    assert "no vale como nombre" in capsys.readouterr().err


def test_nuevo_y_usar(capsys):
    assert main(["nuevo", "cli-a", "--receta", "nano", "--arquitectura", "gpt2", "--reemplazar"]) == 0
    assert main(["nuevo", "cli-a"]) == 1                    # ya existe y no se pidió reemplazar
    assert "Ya existe" in capsys.readouterr().err
    config = Proyecto.abrir("cli-a").config
    assert config["arquitectura"] == "gpt2" and config["modelo"]["posicion"] == "aprendida"
    assert main(["nuevo", "cli-b", "--reemplazar"]) == 0
    assert main(["usar", "cli-a"]) == 0
    assert {"cli-a", "cli-b"} <= set(listar())
    assert main(["proyectos"]) == 0
    assert "► cli-a" in capsys.readouterr().out
    assert main(["generar", "-p", "cli-a"]) == 1            # todavía no hay modelo: se dice qué hacer
    assert f"{ORDEN} entrenar" in capsys.readouterr().err    # la orden, escrita como en este sistema


@pytest.mark.lento
def test_ordenes_sobre_un_proyecto_entrenado(proyecto_mini, capsys, monkeypatch):
    nombre = proyecto_mini.nombre
    activar(nombre)
    assert main(["estado"]) == 0
    salida = capsys.readouterr().out
    assert "✔ 1. Datos" in salida and "✔ 3. Chatbot" in salida

    assert main(["tokenizar", "¿Cuánto es 7 + 8?"]) == 0
    assert "tokens" in capsys.readouterr().out

    assert main(["generar", "Había una vez", "--tokens", "8", "--semilla", "1", "--dispositivo", "cpu", "--etapa", "base"]) == 0
    assert "Había una vez" in capsys.readouterr().out

    assert main(["evaluar", "--casos", "1", "--dispositivo", "cpu"]) == 0
    salida = capsys.readouterr().out
    assert "perplejidad" in salida and "Sumas que NUNCA vio" in salida
    assert (proyecto_mini.carpeta / "evaluacion.json").exists()

    entradas = iter(["Hola", "/dentro", "/nuevo", "/salir"])
    monkeypatch.setattr("builtins.input", lambda _indicador="": next(entradas))
    assert main(["chat", "--tokens", "6", "--dispositivo", "cpu"]) == 0
    salida = capsys.readouterr().out
    assert "cero >" in salida and "[<|usuario|>]" in salida and "conversación nueva" in salida


@pytest.mark.lento
def test_ajustar_pasos_y_reanudar(proyecto_mini, capsys):
    config = proyecto_mini.config
    proyecto = Proyecto.crear("cli-pasos", config, reemplazar=True)
    assert main(["entrenar", "-p", "cli-pasos", "--pasos", "10", "--dispositivo", "cpu"]) == 0
    assert proyecto.config["entreno"]["pasos"] == 10
    assert "Terminado" in capsys.readouterr().out
    # Subir los pasos y reanudar continúa desde donde estaba.
    config = proyecto.config
    config["entreno"]["pasos"] = 14
    proyecto.guardar_config(config)
    assert main(["entrenar", "-p", "cli-pasos", "--reanudar", "--dispositivo", "cpu"]) == 0
    assert "Reanudando desde el paso 10" in capsys.readouterr().out
    assert proyecto.estado()["base"]["paso"] == 14
    assert main(["afinar", "-p", "cli-pasos", "--pasos", "10", "--dispositivo", "cpu"]) == 0
    assert proyecto.estado()["chat_listo"]


@pytest.mark.lento
def test_comprobar_rapido(capsys):
    assert main(["comprobar", "--rapido"]) == 0
    assert "Prueba del modelo: correcta" in capsys.readouterr().out
