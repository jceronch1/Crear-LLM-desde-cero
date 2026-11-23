"""De un montón de texto a los lotes con los que se entrena.

La tarea de un modelo de lenguaje es siempre la misma: **predecir el token siguiente**.
No hace falta que nadie etiquete nada, la respuesta correcta ya está en el propio texto:

    texto (tokens):     Había  una   vez    un   gato
    entrada  x:         Había  una   vez    un
    objetivo y:          una   vez    un   gato        <- lo mismo, desplazado un lugar

Es el «desplazado a la derecha» (shifted right) de la Figura 1 del artículo.

Esta etapa hace cuatro cosas:

    1. leer los documentos del corpus y limpiarlos
    2. apartar una parte para validación (el modelo no la ve nunca mientras aprende)
    3. entrenar el tokenizador, solo con la parte de entrenamiento
    4. convertirlo todo en números y guardarlo en dos archivos binarios

Después, `Lotes` saca de esos archivos ventanas al azar de `contexto` tokens.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import guardar_json, miles, ORDEN
from .cuentos import generar_cuentos
from .proyecto import ErrorDeProyecto, Proyecto, carpeta_descargas
from .tokenizador import INICIO, TokenizadorBPE

# Cuentos infantiles en español: TinyStories traducido (2,1 millones de cuentos en total;
# aquí se usa el primero de sus 8 archivos, con 264.965). Licencia CDLA-Sharing-1.0.
# La dirección apunta a una revisión concreta para que el archivo no cambie nunca.
TINYSTORIES = {
    "url": "https://huggingface.co/datasets/ffuuugor/tinystories_spanish/resolve/"
           "7ba894dc3cb2a314ac77c658f6c315ceba5dd424/data/train-00000-of-00008.parquet",
    "archivo": "tinystories_es-00000.parquet",
    "bytes": 257_960_281,
    "sha256": "72c864aa70dbff7a648a35805697f017d4cdffbeda7a7ee43ee918260651316d",
    "columna": "text_es",
    "ficha": "https://huggingface.co/datasets/ffuuugor/tinystories_spanish",
}

Avisar = Callable[[str], None]

# Un 7,5 % de los cuentos traducidos empieza con una frase del traductor automático
# («Aquí está la traducción al español:»). No es parte del cuento: se quita.
_PREAMBULO = re.compile(
    r"^\s*(aquí (está|tienes) la traducción[^:\n]{0,40}|traducción al español|here is the[^:\n]{0,40}|spanish translation)\s*:\s*",
    re.IGNORECASE,
)


def limpiar(texto: str) -> str:
    """Deja el texto en una forma única: mismos acentos, mismos saltos de línea, sin espacios de más."""
    texto = unicodedata.normalize("NFC", texto).replace("\r\n", "\n").replace("\r", "\n")
    texto = _PREAMBULO.sub("", texto, count=1)
    texto = re.sub(r"[ \t ]+", " ", texto)
    texto = re.sub(r" ?\n ?", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def huella(texto: str) -> int:
    """Un número de 64 bits que identifica el texto. Sirve para quitar duplicados y para repartir."""
    return int.from_bytes(hashlib.blake2b(texto.encode("utf-8"), digest_size=8).digest(), "big")


# ----------------------------------------------------------------------------------------
# Fuentes de documentos
# ----------------------------------------------------------------------------------------
def descargar(url: str, destino: Path, tamano: int | None = None, sha256: str | None = None,
              avisar: Avisar = print) -> Path:
    """Descarga un archivo grande con cuidado: reanuda si se cortó y comprueba que llegó entero."""
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    ya = parcial.stat().st_size if parcial.exists() else 0
    if tamano and ya > tamano:
        parcial.unlink()
        ya = 0
    if not (tamano and ya == tamano):
        cabeceras = {"User-Agent": "cero-llm/1.0"}
        if ya:
            cabeceras["Range"] = f"bytes={ya}-"          # pedir solo lo que falta
        megas = f" ({tamano / 1e6:.0f} MB)" if tamano else ""
        avisar(f"  Descargando {destino.name}{megas}..." + (f" (continúa desde {ya / 1e6:.0f} MB)" if ya else ""))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=cabeceras), timeout=60) as respuesta:
                if ya and respuesta.status != 206:        # el servidor no admite reanudar: empezar de cero
                    ya = 0
                total = ya + int(respuesta.headers.get("Content-Length") or 0)
                ultimo_aviso = time.time()
                with open(parcial, "ab" if ya else "wb") as archivo:
                    while True:
                        trozo = respuesta.read(1 << 20)
                        if not trozo:
                            break
                        archivo.write(trozo)
                        ya += len(trozo)
                        if time.time() - ultimo_aviso > 2:
                            ultimo_aviso = time.time()
                            avisar(f"    {ya / 1e6:6.0f} MB" + (f" de {total / 1e6:.0f} ({100 * ya / total:.0f} %)" if total else ""))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            raise ErrorDeProyecto(
                f"No se pudo descargar el corpus ({error}).\n"
                "  · Comprueba la conexión y repite la orden: la descarga continúa donde se quedó.\n"
                f"  · O descarga el archivo a mano y guárdalo como:  {destino}\n    {url}\n"
                "  · O usa la receta «nano», que no necesita descargar nada."
            ) from error
    if sha256:
        avisar("  Comprobando que el archivo llegó entero...")
        resumen = hashlib.sha256()
        with open(parcial, "rb") as archivo:
            for trozo in iter(lambda: archivo.read(1 << 20), b""):
                resumen.update(trozo)
        if resumen.hexdigest() != sha256:
            parcial.unlink()
            raise ErrorDeProyecto("La descarga llegó dañada (la huella SHA-256 no coincide). Repite la orden para bajarla otra vez.")
    parcial.replace(destino)
    return destino


def _de_tinystories(max_caracteres: int, avisar: Avisar) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ErrorDeProyecto(
            "Para leer el corpus real hace falta el paquete «pyarrow». Instálalo con:\n"
            "    python instalar.py        (lo añade)      o      pip install pyarrow"
        ) from error
    ruta = descargar(TINYSTORIES["url"], carpeta_descargas() / TINYSTORIES["archivo"],
                     TINYSTORIES["bytes"], TINYSTORIES["sha256"], avisar)
    leidos = 0
    for lote in pq.ParquetFile(ruta).iter_batches(batch_size=2048, columns=[TINYSTORIES["columna"]]):
        for texto in lote.column(0).to_pylist():
            if not isinstance(texto, str):
                continue
            texto = limpiar(texto)
            if len(texto) < 100 or "�" in texto:
                continue
            yield texto
            leidos += len(texto)
            if max_caracteres and leidos >= max_caracteres:
                return


def _de_texto(ruta: str) -> Iterator[str]:
    """Tus propios textos: un archivo .txt o una carpeta con varios. Los párrafos se agrupan
    en documentos de unos 1.500 caracteres."""
    origen = Path(ruta).expanduser()
    if not origen.exists():
        raise ErrorDeProyecto(f"No existe {origen}. Indica un archivo .txt o una carpeta con archivos .txt.")
    archivos = [origen] if origen.is_file() else sorted(origen.rglob("*.txt"))
    if not archivos:
        raise ErrorDeProyecto(f"En {origen} no hay archivos .txt.")
    for archivo in archivos:
        contenido = limpiar(archivo.read_text(encoding="utf-8", errors="replace"))
        actual = ""
        for parrafo in contenido.split("\n\n"):
            actual = f"{actual}\n\n{parrafo}" if actual else parrafo
            if len(actual) >= 1500:
                yield actual
                actual = ""
        if len(actual) >= 100:
            yield actual


def documentos(corpus: dict[str, Any], semilla: int = 0, avisar: Avisar = print) -> Iterator[str]:
    """Entrega los documentos del corpus, uno a uno y siempre en el mismo orden."""
    fuente = corpus.get("fuente")
    if fuente == "sintetico":
        yield from generar_cuentos(int(corpus.get("cuentos", 40_000)), semilla, variados=corpus.get("nombres", "mixtos") != "fijos")
    elif fuente == "tinystories":
        yield from _de_tinystories(int(corpus.get("max_caracteres", 0)), avisar)
    elif fuente == "texto":
        yield from _de_texto(corpus["ruta"])
    else:
        raise ErrorDeProyecto(f"Fuente de corpus desconocida: {fuente!r}. Usa «sintetico», «tinystories» o «texto».")


# ----------------------------------------------------------------------------------------
# La etapa completa
# ----------------------------------------------------------------------------------------
def preparar(proyecto: Proyecto, avisar: Avisar = print) -> dict[str, Any]:
    """Corpus -> tokenizador -> `entreno.bin` y `validacion.bin`. Devuelve el manifiesto."""
    config = proyecto.config
    corpus = config["corpus"]
    semilla = int(config["entreno"].get("semilla", 1337))
    fraccion_val = float(corpus.get("validacion", 0.1 if corpus["fuente"] == "texto" else 0.02))
    umbral_val = int(fraccion_val * 10_000)
    inicio = time.time()

    def es_validacion(h: int) -> bool:
        # El reparto depende solo del contenido del documento, no del orden ni del azar:
        # un mismo cuento cae siempre en el mismo lado.
        return h % 10_000 < umbral_val

    # 1. Tokenizador: se aprende con los primeros millones de caracteres DE ENTRENAMIENTO.
    #    (Si viera la validación, estaríamos mirando el examen antes de hacerlo.)
    limite = int(corpus.get("caracteres_tokenizador", 8_000_000))
    avisar(f"[1/3] Leyendo el corpus «{corpus['fuente']}» para entrenar el tokenizador...")
    muestra, acumulado = [], 0
    for texto in documentos(corpus, semilla, avisar):
        if es_validacion(huella(texto)):
            continue
        muestra.append(texto)
        acumulado += len(texto)
        if acumulado >= limite:
            break
    if acumulado < 20_000:
        raise ErrorDeProyecto(
            f"El corpus es demasiado pequeño ({miles(acumulado)} caracteres). Hacen falta al menos unas decenas de miles."
        )
    # También se le enseña una muestra de frases de conversación (preguntas, saludos, cifras):
    # si solo viera cuentos, trocearía en bytes sueltos cosas tan corrientes como «¿» o «7».
    from .chat import textos_para_tokenizador

    muestra.extend(textos_para_tokenizador(corpus, semilla))
    vocab = int(config["vocab"])
    avisar(f"[2/3] Entrenando el tokenizador BPE: vocabulario {miles(vocab)} sobre {acumulado / 1e6:.1f} M de caracteres...")
    t0 = time.time()
    tok = TokenizadorBPE.entrenar(muestra, vocab)
    del muestra
    tok.guardar(proyecto.ruta_tokenizador)
    avisar(f"      {miles(len(tok.fusiones))} fusiones aprendidas en {time.time() - t0:.1f} s. Vocabulario final: {miles(tok.vocab)} tokens.")

    # 2. Convertir todo el corpus en números y repartirlo.
    avisar("[3/3] Convirtiendo el corpus en tokens...")
    proyecto.datos.mkdir(parents=True, exist_ok=True)
    partes = {"entreno": {"documentos": 0, "tokens": 0, "caracteres": 0}, "validacion": {"documentos": 0, "tokens": 0, "caracteres": 0}}
    vistos: set[int] = set()
    duplicados = 0
    ejemplos: list[str] = []
    id_inicio = tok.id(INICIO)
    ultimo_aviso = time.time()
    with open(proyecto.datos / "entreno.bin.tmp", "wb") as f_entreno, open(proyecto.datos / "validacion.bin.tmp", "wb") as f_val:
        for texto in documentos(corpus, semilla, avisar):
            h = huella(texto)
            if h in vistos:
                duplicados += 1
                continue
            vistos.add(h)
            parte = "validacion" if es_validacion(h) else "entreno"
            # Cada documento empieza con <|inicio|>: así el modelo aprende dónde arranca un texto nuevo.
            ids = np.asarray([id_inicio] + tok.codificar_texto(texto), dtype=np.uint16)
            ids.tofile(f_val if parte == "validacion" else f_entreno)
            cuenta = partes[parte]
            cuenta["documentos"] += 1
            cuenta["tokens"] += len(ids)
            cuenta["caracteres"] += len(texto)
            if len(ejemplos) < 5 and parte == "entreno":
                ejemplos.append(texto)
            if time.time() - ultimo_aviso > 3:
                ultimo_aviso = time.time()
                avisar(f"      {partes['entreno']['tokens'] / 1e6:6.1f} M de tokens...")
    (proyecto.datos / "entreno.bin.tmp").replace(proyecto.datos / "entreno.bin")
    (proyecto.datos / "validacion.bin.tmp").replace(proyecto.datos / "validacion.bin")

    contexto = int(config["modelo"]["contexto"])
    for parte, cuenta in partes.items():
        if cuenta["tokens"] < 4 * (contexto + 1):
            raise ErrorDeProyecto(
                f"La parte de {parte} quedó con solo {miles(cuenta['tokens'])} tokens y hacen falta al menos {miles(4 * (contexto + 1))}. "
                "Añade más texto o reduce «contexto» en config.json."
            )

    total_tokens = partes["entreno"]["tokens"] + partes["validacion"]["tokens"]
    total_caracteres = partes["entreno"]["caracteres"] + partes["validacion"]["caracteres"]
    manifiesto = {
        "fuente": corpus["fuente"],
        "vocab": tok.vocab,
        "partes": partes,
        "duplicados_descartados": duplicados,
        "caracteres_por_token": round(total_caracteres / total_tokens, 3),
        "segundos": round(time.time() - inicio, 1),
        "ejemplos": ejemplos,
    }
    if corpus["fuente"] == "tinystories":
        manifiesto["origen"] = TINYSTORIES["ficha"]
        manifiesto["licencia"] = "CDLA-Sharing-1.0"
    guardar_json(proyecto.datos / "manifiesto.json", manifiesto)

    # El modelo necesita saber el tamaño REAL del vocabulario (puede ser menor que el pedido
    # si el corpus es tan pequeño que se queda sin parejas que fusionar).
    config["modelo"]["vocab"] = tok.vocab
    proyecto.guardar_config(config)
    avisar(
        f"      Listo: {miles(partes['entreno']['tokens'])} tokens de entrenamiento y {miles(partes['validacion']['tokens'])} de validación "
        f"({manifiesto['caracteres_por_token']} caracteres por token) en {manifiesto['segundos']} s."
    )
    return manifiesto


# ----------------------------------------------------------------------------------------
# Lotes
# ----------------------------------------------------------------------------------------
class Lotes:
    """Saca lotes (x, y) al azar de un archivo de tokens.

    El archivo no se carga en memoria: `np.memmap` lo deja en el disco y lee solo lo que hace
    falta. Así un corpus de cientos de megas funciona igual en un portátil con poca RAM.
    """

    def __init__(self, ruta: Path | str, contexto: int, lote: int, semilla: int, dispositivo: str = "cpu") -> None:
        ruta = Path(ruta)
        if not ruta.exists():
            raise ErrorDeProyecto(f"Falta {ruta}. Ejecuta antes:  {ORDEN} datos")
        self.tokens = np.memmap(ruta, dtype=np.uint16, mode="r")
        if len(self.tokens) < contexto + 2:
            raise ErrorDeProyecto(f"{ruta.name} solo tiene {miles(len(self.tokens))} tokens: no cabe ni una ventana de {contexto}.")
        self.contexto, self.lote, self.dispositivo = contexto, lote, dispositivo
        self.azar = np.random.default_rng(semilla)
        self._ventana = np.arange(contexto + 1)

    def siguiente(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Un lote: x e y tienen forma (lote, contexto); y es x desplazado un token."""
        inicios = self.azar.integers(0, len(self.tokens) - self.contexto - 1, size=self.lote)
        trozos = torch.from_numpy(self.tokens[inicios[:, None] + self._ventana].astype(np.int64))
        x, y = trozos[:, :-1], trozos[:, 1:]
        if self.dispositivo == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.dispositivo), y.to(self.dispositivo)

    # Para poder reanudar un entrenamiento exactamente donde se quedó.
    def estado(self) -> dict[str, Any]:
        return self.azar.bit_generator.state

    def restaurar(self, estado: dict[str, Any]) -> None:
        self.azar.bit_generator.state = estado
