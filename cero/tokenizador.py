"""Del texto a los números: un tokenizador BPE a nivel de bytes, escrito desde cero.

Un modelo de lenguaje no ve letras: ve una lista de números enteros (*tokens*). Este archivo
aprende qué trozos de texto merecen un número propio. Es el mismo algoritmo (Byte Pair
Encoding) que usan GPT-2, GPT-4 y casi todos los LLM actuales, y el que cita el artículo
«Attention Is All You Need» en su sección 5.1.

La idea, en cuatro pasos:

1. El texto se corta en trozos con una expresión regular (palabras, números, signos), para
   que una fusión nunca pegue el final de una palabra con el principio de la siguiente.
2. Cada trozo se escribe como sus bytes UTF-8. El vocabulario inicial son los 256 valores
   posibles de un byte, así que CUALQUIER texto se puede codificar: no existe el «token
   desconocido». (Una «ñ» o una «á» ocupan 2 bytes; el algoritmo aprende enseguida a unirlos.)
3. Entrenar = repetir: buscar la pareja de tokens vecinos más frecuente y fusionarla en un
   token nuevo. Tras `vocab - 256` fusiones, las palabras frecuentes tienen su propio número.
4. Codificar = volver a aplicar esas fusiones en el mismo orden. Decodificar = pegar los
   bytes de cada token.

Además reservamos cuatro *tokens especiales* que nunca salen del texto normal y que
convertirán el modelo en un chatbot (ver cero/chat.py):

    <|inicio|>     empieza un documento o una conversación
    <|usuario|>    a continuación habla la persona
    <|asistente|>  a continuación habla el modelo
    <|fin|>        termina un turno (es la señal para dejar de generar)
"""

from __future__ import annotations

import codecs
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

INICIO, USUARIO, ASISTENTE, FIN = "<|inicio|>", "<|usuario|>", "<|asistente|>", "<|fin|>"
ESPECIALES: tuple[str, ...] = (INICIO, USUARIO, ASISTENTE, FIN)

# Cómo se corta el texto antes de aplicar BPE. De izquierda a derecha:
#   una palabra (con su espacio delante, si lo hay)  ·  un número de hasta 3 cifras
#   signos de puntuación seguidos  ·  espacios y saltos de línea  ·  guiones bajos
# Entre todas las alternativas cubren cualquier carácter: no se pierde nada del texto.
PATRON = r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+|_+"

Par = tuple[int, int]


def fusionar(ids: Sequence[int], par: Par, nuevo: int) -> list[int]:
    """Sustituye cada aparición de `par` en `ids` por el token `nuevo` (de izquierda a derecha)."""
    a, b = par
    salida: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            salida.append(nuevo)
            i += 2
        else:
            salida.append(ids[i])
            i += 1
    return salida


class TokenizadorBPE:
    """BPE a nivel de bytes. Se crea con `TokenizadorBPE.entrenar(...)` o con `.cargar(ruta)`."""

    def __init__(self, fusiones: Sequence[Par], especiales: Sequence[str] = ESPECIALES, patron: str = PATRON) -> None:
        self.fusiones: list[Par] = [(int(a), int(b)) for a, b in fusiones]
        self.patron = patron
        self._cortar = re.compile(patron)
        # rango[par] = en qué momento se aprendió esa fusión (0 = la primera, la más frecuente)
        self._rango: dict[Par, int] = {par: i for i, par in enumerate(self.fusiones)}
        # Los bytes que representa cada token: primero los 256 bytes sueltos, luego cada fusión.
        self._bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.fusiones:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        # Los especiales ocupan los últimos números del vocabulario.
        primero = len(self._bytes)
        self.especiales: dict[str, int] = {texto: primero + i for i, texto in enumerate(especiales)}
        self._especial_por_id = {i: texto for texto, i in self.especiales.items()}
        self._separar_especiales = (
            re.compile("(" + "|".join(re.escape(t) for t in self.especiales) + ")") if self.especiales else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ propiedades
    @property
    def vocab(self) -> int:
        """Cuántos tokens distintos existen: 256 bytes + fusiones + especiales."""
        return len(self._bytes) + len(self.especiales)

    def id(self, especial: str) -> int:
        """El número de un token especial, por ejemplo `tok.id("<|fin|>")`."""
        return self.especiales[especial]

    # ------------------------------------------------------------------ entrenar
    @classmethod
    def entrenar(
        cls,
        textos: str | Iterable[str],
        vocab: int,
        especiales: Sequence[str] = ESPECIALES,
        patron: str = PATRON,
        avisar: Callable[[int, int], None] | None = None,
    ) -> "TokenizadorBPE":
        """Aprende `vocab - 256 - len(especiales)` fusiones a partir de `textos`.

        Es el algoritmo de siempre, con dos trucos para que tarde segundos y no horas:

        * Cada trozo distinto se guarda UNA vez junto con cuántas veces aparece. Fusionar una
          pareja dentro de « gato» cuesta lo mismo si la palabra sale 3 veces o 30.000.
        * No se recuentan todas las parejas en cada vuelta: solo se actualizan las cuentas de
          los trozos que contenían la pareja fusionada. Un montículo (heap) entrega siempre
          la pareja más frecuente.

        El resultado es idéntico al del algoritmo «ingenuo» que se escribe en la clase 1 del
        curso (hay una prueba que lo comprueba). En caso de empate gana la pareja menor.
        """
        n_fusiones = vocab - 256 - len(especiales)
        if n_fusiones < 0:
            raise ValueError(f"El vocabulario mínimo es {256 + len(especiales)} (256 bytes + {len(especiales)} especiales).")
        cortar = re.compile(patron)
        quitar = re.compile("|".join(re.escape(t) for t in especiales)) if especiales else None

        # 1. Contar cada trozo distinto. Los tokens especiales se recortan antes: nunca se fusionan.
        cuenta: Counter[str] = Counter()
        for texto in [textos] if isinstance(textos, str) else textos:
            for parte in quitar.split(texto) if quitar else [texto]:
                cuenta.update(cortar.findall(parte))
        palabras = [list(trozo.encode("utf-8")) for trozo in cuenta]
        veces = list(cuenta.values())

        # 2. Contar las parejas vecinas (pesadas por la frecuencia del trozo) y anotar dónde están.
        cuenta_pares: defaultdict[Par, int] = defaultdict(int)
        donde: defaultdict[Par, set[int]] = defaultdict(set)
        for i, palabra in enumerate(palabras):
            for par in zip(palabra, palabra[1:]):
                cuenta_pares[par] += veces[i]
                donde[par].add(i)
        monton = [(-n, par) for par, n in cuenta_pares.items()]
        heapq.heapify(monton)

        # 3. Fusionar la pareja más frecuente, actualizar solo lo afectado, y repetir.
        fusiones: list[Par] = []
        while len(fusiones) < n_fusiones and monton:
            menos_n, par = heapq.heappop(monton)
            real = cuenta_pares.get(par, 0)
            if real <= 0:
                continue                                    # esa pareja ya no existe
            if -menos_n != real:                            # anotación antigua: se corrige y se devuelve
                heapq.heappush(monton, (-real, par))
                continue
            nuevo = 256 + len(fusiones)
            fusiones.append(par)
            tocados: set[Par] = set()
            for i in donde.pop(par, ()):
                antes = palabras[i]
                despues = fusionar(antes, par, nuevo)
                if len(despues) == len(antes):
                    continue
                for p in zip(antes, antes[1:]):
                    cuenta_pares[p] -= veces[i]
                for p in zip(despues, despues[1:]):
                    cuenta_pares[p] += veces[i]
                    donde[p].add(i)
                    tocados.add(p)
                palabras[i] = despues
            cuenta_pares.pop(par, None)
            for p in tocados:
                if cuenta_pares.get(p, 0) > 0:
                    heapq.heappush(monton, (-cuenta_pares[p], p))
            if avisar is not None:
                avisar(len(fusiones), n_fusiones)
        return cls(fusiones, especiales=especiales, patron=patron)

    # ------------------------------------------------------------------ codificar
    def _bpe(self, trozo: str) -> list[int]:
        """Aplica a un trozo las fusiones aprendidas, de la más antigua a la más reciente."""
        ids = list(trozo.encode("utf-8"))
        sin_rango = len(self._rango)
        while len(ids) >= 2:
            mejor = min(zip(ids, ids[1:]), key=lambda p: self._rango.get(p, sin_rango))
            rango = self._rango.get(mejor)
            if rango is None:
                break                                       # ya no queda ninguna pareja conocida
            ids = fusionar(ids, mejor, 256 + rango)
        if len(self._cache) < 1_000_000:
            self._cache[trozo] = ids
        return ids

    def codificar_texto(self, texto: str) -> list[int]:
        """Texto normal -> tokens. Aquí «<|fin|>» sería solo una ristra de caracteres corrientes.

        Es la función que hay que usar con lo que escribe una persona: así nadie puede colar
        un token especial dentro de su mensaje y hacerse pasar por el asistente.
        """
        cache, salida = self._cache, []
        for trozo in self._cortar.findall(texto):
            ids = cache.get(trozo)
            salida.extend(ids if ids is not None else self._bpe(trozo))
        return salida

    def codificar(self, texto: str) -> list[int]:
        """Texto -> tokens, reconociendo los tokens especiales que aparezcan escritos."""
        if self._separar_especiales is None:
            return self.codificar_texto(texto)
        salida: list[int] = []
        for parte in self._separar_especiales.split(texto):
            if parte in self.especiales:
                salida.append(self.especiales[parte])
            elif parte:
                salida.extend(self.codificar_texto(parte))
        return salida

    # ------------------------------------------------------------------ decodificar
    def bytes_de(self, token: int) -> bytes:
        """Los bytes exactos que representa un token."""
        token = int(token)
        if token in self._especial_por_id:
            return self._especial_por_id[token].encode("utf-8")
        if 0 <= token < len(self._bytes):
            return self._bytes[token]
        return b""                                          # número fuera del vocabulario: se ignora

    def decodificar(self, ids: Iterable[int]) -> str:
        """Tokens -> texto. Los bytes que no formen un carácter válido se muestran como «�»."""
        return b"".join(self.bytes_de(i) for i in ids).decode("utf-8", errors="replace")

    def texto_de(self, token: int) -> str:
        """Un token suelto, para mostrarlo. Un token puede ser medio carácter (p. ej. un byte de «ñ»)."""
        return self.bytes_de(token).decode("utf-8", errors="replace")

    def piezas(self, texto: str, especiales: bool = True) -> list[dict[str, Any]]:
        """El texto troceado, con el número, el texto y los bytes de cada token (para visualizarlo)."""
        ids = self.codificar(texto) if especiales else self.codificar_texto(texto)
        return [
            {"id": i, "texto": self.texto_de(i), "bytes": list(self.bytes_de(i)), "especial": i in self._especial_por_id}
            for i in ids
        ]

    def flujo(self) -> "DecodificadorIncremental":
        """Un decodificador para ir mostrando el texto token a token mientras el modelo escribe."""
        return DecodificadorIncremental(self)

    # ------------------------------------------------------------------ guardar y cargar
    def a_dict(self) -> dict[str, Any]:
        return {
            "tipo": "bpe-bytes",
            "version": 1,
            "patron": self.patron,
            "fusiones": [list(par) for par in self.fusiones],
            "especiales": list(self.especiales),
        }

    @classmethod
    def desde_dict(cls, datos: dict[str, Any]) -> "TokenizadorBPE":
        if datos.get("tipo") != "bpe-bytes":
            raise ValueError("Ese archivo no describe un tokenizador de este proyecto.")
        return cls([tuple(par) for par in datos["fusiones"]], especiales=datos["especiales"], patron=datos["patron"])

    def guardar(self, ruta: Path | str) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(self.a_dict()), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path | str) -> "TokenizadorBPE":
        return cls.desde_dict(json.loads(Path(ruta).read_text(encoding="utf-8")))


class DecodificadorIncremental:
    """Convierte tokens en texto de uno en uno sin romper los caracteres de varios bytes.

    La «ó» son dos bytes. Si el modelo los emite en dos tokens distintos, el primero solo no
    se puede mostrar todavía: este objeto lo guarda y lo entrega cuando llega el segundo.
    """

    def __init__(self, tok: TokenizadorBPE) -> None:
        self._tok = tok
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def agregar(self, token: int) -> str:
        return self._utf8.decode(self._tok.bytes_de(token))

    def terminar(self) -> str:
        return self._utf8.decode(b"", final=True)
