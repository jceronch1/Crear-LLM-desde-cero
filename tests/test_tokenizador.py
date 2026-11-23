"""El tokenizador: no pierde nada del texto y el algoritmo rápido da lo mismo que el de libro."""

from __future__ import annotations

import random
import re
from collections import Counter

from cero.tokenizador import ESPECIALES, FIN, PATRON, TokenizadorBPE, fusionar

TEXTOS = [
    "Había una vez una niña pequeña llamada Lucía. ¡Qué alegría!",
    "El ñandú comió 1234 uvas_verdes — «¿de verdad?» —preguntó.\n\n\tFin.   ",
    "emoji 🙂, árabe مرحبا, chino 你好, símbolos ±½²",
    "",
    "   ",
    "a",
]


def bpe_de_libro(texto: str, n_fusiones: int) -> list[tuple[int, int]]:
    """El algoritmo ingenuo, tal cual se explica en la clase 1: recontar todo en cada vuelta."""
    palabras = Counter(re.findall(PATRON, texto))
    trozos = {palabra: list(palabra.encode("utf-8")) for palabra in palabras}
    fusiones: list[tuple[int, int]] = []
    for _ in range(n_fusiones):
        pares: Counter[tuple[int, int]] = Counter()
        for palabra, ids in trozos.items():
            for par in zip(ids, ids[1:]):
                pares[par] += palabras[palabra]
        if not pares:
            break
        mejor = min(pares, key=lambda par: (-pares[par], par))       # el más frecuente; en empate, el menor
        nuevo = 256 + len(fusiones)
        fusiones.append(mejor)
        trozos = {palabra: fusionar(ids, mejor, nuevo) for palabra, ids in trozos.items()}
    return fusiones


def test_el_patron_no_pierde_ningun_caracter():
    azar = random.Random(0)
    alfabeto = "abc áéñü ÁÑ 0123456789 _-—.,;:¡!¿?()\"'«»\n\t🙂你好½²"
    for _ in range(300):
        texto = "".join(azar.choice(alfabeto) for _ in range(azar.randint(0, 60)))
        assert "".join(re.findall(PATRON, texto)) == texto


def test_ida_y_vuelta(tokenizador):
    for texto in TEXTOS:
        assert tokenizador.decodificar(tokenizador.codificar(texto)) == texto
        assert tokenizador.decodificar(tokenizador.codificar_texto(texto)) == texto


def test_el_algoritmo_rapido_coincide_con_el_de_libro():
    from cero.cuentos import generar_cuentos

    texto = " ".join(generar_cuentos(60, 3)) + " aaaa aaa aa a ¿¿¡¡ ññ"
    rapido = TokenizadorBPE.entrenar(texto, 256 + 180 + len(ESPECIALES))
    assert rapido.fusiones == bpe_de_libro(texto, 180)


def test_tamano_del_vocabulario(tokenizador):
    assert tokenizador.vocab == 256 + len(tokenizador.fusiones) + len(ESPECIALES)
    assert sorted(tokenizador.especiales.values()) == list(range(tokenizador.vocab - 4, tokenizador.vocab))


def test_comprime(tokenizador):
    texto = "Había una vez un gato pequeño que vivía en el bosque."
    assert len(tokenizador.codificar(texto)) < len(texto.encode("utf-8")) / 2


def test_tokens_especiales(tokenizador):
    ids = tokenizador.codificar(f"hola{FIN}")
    assert ids[-1] == tokenizador.id(FIN)
    # Lo que escribe una persona nunca se convierte en token especial.
    literal = tokenizador.codificar_texto(f"hola{FIN}")
    assert tokenizador.id(FIN) not in literal
    assert tokenizador.decodificar(literal) == f"hola{FIN}"


def test_guardar_y_cargar(tokenizador, tmp_path):
    ruta = tmp_path / "tok.json"
    tokenizador.guardar(ruta)
    copia = TokenizadorBPE.cargar(ruta)
    texto = "Lucía encontró una llave dorada."
    assert copia.codificar(texto) == tokenizador.codificar(texto)
    assert copia.vocab == tokenizador.vocab
    assert TokenizadorBPE.desde_dict(tokenizador.a_dict()).fusiones == tokenizador.fusiones


def test_decodificador_incremental(tokenizador):
    texto = "¡Qué año tan señalado, niña! 🙂"
    flujo = tokenizador.flujo()
    trozos = [flujo.agregar(i) for i in tokenizador.codificar(texto)]
    assert "".join(trozos) + flujo.terminar() == texto
    assert all("�" not in trozo for trozo in trozos)        # nunca medio carácter en pantalla


def test_piezas(tokenizador):
    piezas = tokenizador.piezas("Hola<|fin|>")
    assert piezas[-1]["especial"] and piezas[-1]["texto"] == "<|fin|>"
    assert b"".join(bytes(p["bytes"]) for p in piezas).decode("utf-8") == "Hola<|fin|>"


def test_vocabulario_demasiado_pequeno():
    import pytest

    with pytest.raises(ValueError):
        TokenizadorBPE.entrenar("hola", 100)
