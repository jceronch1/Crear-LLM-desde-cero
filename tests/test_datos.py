"""Los datos: limpieza, cuentos sintéticos, la etapa completa y los lotes."""

from __future__ import annotations

import random
import re

import numpy as np
import pytest

from cero import datos
from cero.cuentos import RESERVAS, TRAMAS, cuento, generar_cuentos, nombre_inventado
from cero.config import ORDEN
from cero.proyecto import ErrorDeProyecto, Proyecto

RESERVADOS = re.compile(r"\b(" + "|".join(n for listas in RESERVAS.values() for lista in listas for n in lista) + r")\b")


def test_limpiar():
    sucio = "Aquí está la traducción al español:\r\n\r\nHabía   una vez\tun gato.  \n\n\n\nFin. "
    assert datos.limpiar(sucio) == "Había una vez un gato.\n\nFin."
    assert datos.limpiar("Había una vez, aquí está la traducción: no se toca") == "Había una vez, aquí está la traducción: no se toca"


def test_los_cuentos_son_repetibles_y_variados():
    assert list(generar_cuentos(50, 7)) == list(generar_cuentos(50, 7))
    assert list(generar_cuentos(50, 7)) != list(generar_cuentos(50, 8))
    assert len(set(generar_cuentos(2000, 0))) == 2000


def test_todas_las_tramas_y_la_concordancia():
    azar = random.Random(0)
    for trama in TRAMAS:
        for _ in range(40):
            c = cuento(azar, trama=trama)
            p = c.protagonista
            assert p.nombre in c.texto
            if trama == "tesoro":
                assert c.amigo is None                      # es la única trama con un solo personaje
            else:
                assert c.amigo.nombre in c.texto
                assert p.tipo != c.amigo.tipo and p.nombre != c.amigo.nombre
            assert f"{'un' if p.genero == 'm' else 'una'} {p.tipo} {p.adjetivo}" in c.texto
            # Ningún artículo masculino pegado a un personaje femenino, ni al revés.
            assert f"{'una' if p.genero == 'm' else 'un'} {p.tipo} " not in c.texto
            assert all(pregunta.startswith("¿") and respuesta.endswith(".") for pregunta, respuesta in c.hechos)


def test_los_nombres_reservados_no_aparecen_nunca_al_entrenar():
    assert not any(RESERVADOS.search(texto) for texto in generar_cuentos(6000, 3))
    azar = random.Random(0)
    assert not any(RESERVADOS.fullmatch(nombre_inventado(azar)) for _ in range(200_000))
    # ...y sí aparecen cuando se piden para evaluar.
    for reserva, (masculinos, femeninos) in RESERVAS.items():
        assert cuento(azar, reserva=reserva).protagonista.nombre in masculinos + femeninos


def test_nombres_fijos_o_variados():
    fijos = " ".join(generar_cuentos(300, 1, variados=False))
    variados = " ".join(generar_cuentos(300, 1, variados=True))
    nombres = lambda texto: set(re.findall(r"llamad[oa] ([A-ZÁÉÍÓÚ]\w+)", texto))  # noqa: E731
    assert len(nombres(variados)) > len(nombres(fijos))


@pytest.mark.lento
def test_la_etapa_de_datos(proyecto_mini):
    estado = proyecto_mini.estado()["datos"]
    tok = proyecto_mini.tokenizador()
    assert estado["vocab"] == tok.vocab == proyecto_mini.config["modelo"]["vocab"]
    entreno = np.memmap(proyecto_mini.datos / "entreno.bin", dtype=np.uint16, mode="r")
    validacion = np.memmap(proyecto_mini.datos / "validacion.bin", dtype=np.uint16, mode="r")
    assert len(entreno) == estado["partes"]["entreno"]["tokens"]
    assert len(validacion) == estado["partes"]["validacion"]["tokens"]
    assert int(entreno.max()) < tok.vocab
    assert entreno[0] == tok.id("<|inicio|>")
    assert int((entreno == tok.id("<|inicio|>")).sum()) == estado["partes"]["entreno"]["documentos"]
    # Lo que se guardó se puede leer otra vez como texto.
    primero = tok.decodificar(entreno[1:40].tolist())
    assert estado["ejemplos"][0].startswith(primero[:20])
    # Ningún cuento está a la vez en entrenamiento y en validación.
    def cuentos(tokens):
        return set(tok.decodificar(tokens.tolist()).split("<|inicio|>")) - {""}
    assert not (cuentos(entreno) & cuentos(validacion))


@pytest.mark.lento
def test_los_lotes(proyecto_mini):
    lotes = datos.Lotes(proyecto_mini.datos / "entreno.bin", contexto=16, lote=4, semilla=5)
    estado = lotes.estado()
    x, y = lotes.siguiente()
    assert x.shape == y.shape == (4, 16)
    assert (x[:, 1:] == y[:, :-1]).all()                    # y es x desplazado un token
    otro_x, _ = lotes.siguiente()
    assert not (otro_x == x).all()
    lotes.restaurar(estado)                                 # con el mismo estado sale el mismo lote
    repetido, _ = lotes.siguiente()
    assert (repetido == x).all()


def test_textos_propios(tmp_path):
    from conftest import receta_mini

    carpeta = tmp_path / "mis_textos"
    carpeta.mkdir()
    for i in range(3):
        (carpeta / f"libro{i}.txt").write_text("\n\n".join(generar_cuentos(150, i)), encoding="utf-8")
    config = receta_mini()
    config["corpus"] = {"fuente": "texto", "ruta": str(carpeta)}
    config["modelo"]["contexto"] = 64
    proyecto = Proyecto.crear("textos-propios", config, reemplazar=True)
    manifiesto = datos.preparar(proyecto, lambda _m: None)
    assert manifiesto["fuente"] == "texto"
    assert manifiesto["partes"]["validacion"]["tokens"] > 0


def test_errores_claros(tmp_path):
    from conftest import receta_mini

    config = receta_mini()
    config["corpus"] = {"fuente": "texto", "ruta": str(tmp_path / "no_existe")}
    proyecto = Proyecto.crear("sin-textos", config, reemplazar=True)
    with pytest.raises(ErrorDeProyecto, match="No existe"):
        datos.preparar(proyecto, lambda _m: None)
    with pytest.raises(ErrorDeProyecto, match=re.escape(f"{ORDEN} datos")):
        datos.Lotes(proyecto.datos / "entreno.bin", 16, 4, 0)
    with pytest.raises(ErrorDeProyecto, match="no vale como nombre"):
        Proyecto("con espacios")
