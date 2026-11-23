"""Toda la configuración en un solo lugar: el modelo, el entrenamiento y las recetas.

Una *receta* es un proyecto completo ya dimensionado (corpus + tokenizador + modelo +
entrenamiento + ajuste para chat). Hay cuatro, de menor a mayor:

    nano    ~1 M de parámetros   cuentos sintéticos (sin descargas)   minutos en CPU
    micro   ~5 M                 cuentos reales en español            CPU paciente o GPU
    mini    ~15 M                cuentos reales en español            GPU (minutos)
    base    ~35 M                cuentos reales en español            GPU (media hora)

Los nombres de los ajustes están en español, pero al lado de cada uno va el nombre que
verás en los artículos y en otros repositorios (d_model, n_layers, ...).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


# Cómo se escribe la orden en la consola de este sistema, desde la carpeta del proyecto. Los
# mensajes la usan para que se puedan copiar tal cual: «.\cero chat» en Windows (PowerShell y
# CMD) y «./cero.sh chat» en Linux y macOS. (Con el entorno activado, también vale «cero chat».)
ORDEN = r".\cero" if os.name == "nt" else "./cero.sh"


class ErrorDeConfig(ValueError):
    """Un ajuste imposible. El mensaje dice cuál y cómo arreglarlo."""


# --------------------------------------------------------------------------------------
# El modelo
# --------------------------------------------------------------------------------------
@dataclass
class ConfigModelo:
    vocab: int = 1024          # tamaño del vocabulario (vocab_size)
    contexto: int = 128        # tokens que el modelo ve a la vez (context length / block size)
    d_modelo: int = 128        # ancho de cada vector (d_model / n_embd)
    n_capas: int = 4           # bloques Transformer apilados (N / n_layers)
    n_cabezas: int = 4         # cabezas de atención por bloque (h / n_heads)
    # --- Interruptores de arquitectura: cada uno es una idea de la historia del Transformer.
    posicion: str = "rope"     # "sinusoidal" (artículo 2017) | "aprendida" (GPT-2) | "rope" (moderno)
    norma: str = "rms"         # "capa" = LayerNorm (2017, GPT-2) | "rms" = RMSNorm (moderno)
    pre_norma: bool = True     # False: norma DESPUÉS de sumar (2017). True: norma ANTES (GPT-2 en adelante)
    mlp: str = "swiglu"        # "relu" (2017) | "gelu" (GPT-2) | "swiglu" (moderno)
    atar_pesos: bool = True    # la tabla de entrada y la capa de salida comparten matriz (artículo, §3.4)
    sesgos: bool = False       # términos de sesgo (bias) en las capas lineales
    dropout: float = 0.0       # apagar neuronas al azar al entrenar (el artículo usa 0.1)

    def __post_init__(self) -> None:
        def exige(condicion: bool, mensaje: str) -> None:
            if not condicion:
                raise ErrorDeConfig(f"Configuración del modelo: {mensaje}")

        for nombre in ("vocab", "contexto", "d_modelo", "n_capas", "n_cabezas"):
            exige(isinstance(getattr(self, nombre), int) and getattr(self, nombre) > 0,
                  f"«{nombre}» debe ser un entero positivo (vale {getattr(self, nombre)!r}).")
        exige(self.d_modelo % self.n_cabezas == 0,
              f"d_modelo ({self.d_modelo}) debe ser divisible entre n_cabezas ({self.n_cabezas}): "
              "cada cabeza recibe d_modelo / n_cabezas dimensiones.")
        exige(self.d_cabeza % 2 == 0,
              f"la dimensión por cabeza ({self.d_cabeza}) debe ser par: RoPE y la codificación "
              "sinusoidal trabajan con parejas de números.")
        exige(self.posicion in ("sinusoidal", "aprendida", "rope"),
              f"posicion={self.posicion!r} no existe; usa 'sinusoidal', 'aprendida' o 'rope'.")
        exige(self.norma in ("capa", "rms"), f"norma={self.norma!r} no existe; usa 'capa' o 'rms'.")
        exige(self.mlp in ("relu", "gelu", "swiglu"),
              f"mlp={self.mlp!r} no existe; usa 'relu', 'gelu' o 'swiglu'.")
        exige(0.0 <= self.dropout < 1.0, f"dropout debe estar en [0, 1) (vale {self.dropout}).")
        exige(self.vocab <= 65536, "el vocabulario no puede pasar de 65.536 (los tokens se guardan en 16 bits).")

    @property
    def d_cabeza(self) -> int:
        """Dimensiones de cada cabeza de atención (d_k en el artículo)."""
        return self.d_modelo // self.n_cabezas

    @property
    def d_oculta(self) -> int:
        """Ancho interior de la red de avance (d_ff en el artículo).

        ReLU y GELU usan 4 x d_modelo (el artículo: 512 -> 2048). SwiGLU tiene tres matrices
        en lugar de dos, así que usa 8/3 x d_modelo para gastar los mismos parámetros.
        """
        if self.mlp == "swiglu":
            return 8 * round(8 * self.d_modelo / 3 / 8)
        return 4 * self.d_modelo

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, valores: dict[str, Any]) -> "ConfigModelo":
        conocidos = {f.name for f in fields(cls)}
        sobrantes = sorted(set(valores) - conocidos)
        if sobrantes:
            raise ErrorDeConfig(f"Ajustes de modelo desconocidos: {sobrantes}. Los válidos son: {sorted(conocidos)}")
        return cls(**valores)


# Tres generaciones de la misma idea. Cambian solo los interruptores; el resto del código es igual.
ARQUITECTURAS: dict[str, dict[str, Any]] = {
    # «Attention Is All You Need» (2017), lado decodificador: posiciones con senos y cosenos,
    # LayerNorm después de cada suma, red de avance con ReLU y dropout 0.1.
    "paper2017": dict(posicion="sinusoidal", norma="capa", pre_norma=False, mlp="relu", sesgos=True, dropout=0.1),
    # GPT-2 (2019): posiciones aprendidas, la norma pasa ANTES de cada subcapa, GELU.
    "gpt2": dict(posicion="aprendida", norma="capa", pre_norma=True, mlp="gelu", sesgos=True, dropout=0.0),
    # Llama, Qwen, Gemma... (2023 en adelante): RoPE, RMSNorm, SwiGLU, sin sesgos.
    "moderna": dict(posicion="rope", norma="rms", pre_norma=True, mlp="swiglu", sesgos=False, dropout=0.0),
}


# --------------------------------------------------------------------------------------
# El entrenamiento
# --------------------------------------------------------------------------------------
@dataclass
class ConfigEntreno:
    pasos: int = 600                 # actualizaciones de los pesos
    lote: int = 32                   # secuencias por paso (batch size)
    acumular: int = 1                # pasos de gradiente acumulados (simula un lote mayor)
    lr: float = 3e-3                 # tasa de aprendizaje máxima (learning rate)
    lr_final: float = 0.1            # fracción de lr a la que se llega al final del coseno
    calentamiento: int = 40          # pasos de subida lineal de la tasa (warmup)
    decaimiento: float = 0.1         # decaimiento de pesos (weight decay), solo en matrices
    recorte: float = 1.0             # norma máxima del gradiente (gradient clipping)
    evaluar_cada: int = 100          # cada cuántos pasos se mide la validación
    lotes_eval: int = 16             # lotes de validación por medición
    semilla: int = 1337

    def __post_init__(self) -> None:
        for nombre in ("pasos", "lote", "acumular", "evaluar_cada", "lotes_eval"):
            if int(getattr(self, nombre)) < 1:
                raise ErrorDeConfig(f"Entrenamiento: «{nombre}» debe ser al menos 1.")
        if not 0 < self.lr < 1:
            raise ErrorDeConfig(f"Entrenamiento: lr={self.lr} no es razonable; prueba entre 1e-4 y 1e-2.")
        if self.calentamiento >= self.pasos:
            self.calentamiento = max(1, self.pasos // 10)

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, valores: dict[str, Any]) -> "ConfigEntreno":
        conocidos = {f.name for f in fields(cls)}
        sobrantes = sorted(set(valores) - conocidos)
        if sobrantes:
            raise ErrorDeConfig(f"Ajustes de entrenamiento desconocidos: {sobrantes}. Los válidos son: {sorted(conocidos)}")
        return cls(**valores)


# --------------------------------------------------------------------------------------
# Las recetas
# --------------------------------------------------------------------------------------
# «corpus» es de dónde sale el texto:
#   sintetico    cuentos generados por reglas (cero/cuentos.py). No descarga nada.
#   tinystories  cuentos infantiles reales en español (Hugging Face, 258 MB la primera vez).
#   texto        tus propios archivos .txt  (cero nuevo ... --texto carpeta_o_archivo)
#
# Los tiempos de las descripciones se midieron en un portátil con Ryzen 7 8845HS y RTX 4070
# (8 GB). En cada equipo son distintos: `cero comprobar` los estima para el tuyo.
RECETAS: dict[str, dict[str, Any]] = {
    "nano": {
        "descripcion": "El primer modelo: ~1 M de parámetros con cuentos sintéticos, sin descargas. Unos 6 minutos en CPU y 2 en GPU.",
        "corpus": {"fuente": "sintetico", "cuentos": 40_000, "nombres": "mixtos"},
        "vocab": 1024,
        "modelo": dict(contexto=256, d_modelo=128, n_capas=4, n_cabezas=4),
        # Lotes pequeños y muchos pasos: con el mismo número de tokens, el modelo aprende más
        # (medido: misma duración, mejores resultados que con lotes de 16 y 32).
        "entreno": dict(pasos=1500, lote=8, lr=3e-3, calentamiento=60, evaluar_cada=150),
        "chat": dict(pasos=2000, lote=16, lr=1e-3, calentamiento=40, evaluar_cada=250, decaimiento=0.0),
        "dialogos": 40_000,
    },
    "micro": {
        "descripcion": "~6 M de parámetros con cuentos reales en español. Unos 3 minutos en GPU; alrededor de una hora en CPU.",
        "corpus": {"fuente": "tinystories", "max_caracteres": 40_000_000},
        "vocab": 4096,
        "modelo": dict(contexto=256, d_modelo=256, n_capas=6, n_cabezas=4),
        "entreno": dict(pasos=3000, lote=32, lr=2e-3, calentamiento=100, evaluar_cada=250),
        "chat": dict(pasos=1000, lote=32, lr=5e-4, calentamiento=40, evaluar_cada=200, decaimiento=0.0),
        "dialogos": 40_000,
    },
    "mini": {
        "descripcion": "~16 M de parámetros con cuentos reales. Unos 20 minutos en GPU; no es práctica en CPU.",
        "corpus": {"fuente": "tinystories", "max_caracteres": 120_000_000},
        "vocab": 4096,
        "modelo": dict(contexto=256, d_modelo=384, n_capas=8, n_cabezas=6),
        "entreno": dict(pasos=5000, lote=64, lr=1.5e-3, calentamiento=200, evaluar_cada=500),
        "chat": dict(pasos=1500, lote=64, lr=3e-4, calentamiento=60, evaluar_cada=250, decaimiento=0.0),
        "dialogos": 60_000,
    },
    "base": {
        "descripcion": "~36 M de parámetros con todo el corpus. Solo GPU: unos 50 minutos en una de 8 GB (usa casi 7).",
        "corpus": {"fuente": "tinystories", "max_caracteres": 0},
        "vocab": 8192,
        "modelo": dict(contexto=512, d_modelo=512, n_capas=10, n_cabezas=8),
        "entreno": dict(pasos=7000, lote=32, lr=1e-3, calentamiento=300, evaluar_cada=500),
        "chat": dict(pasos=1500, lote=32, lr=2e-4, calentamiento=60, evaluar_cada=250, decaimiento=0.0),
        "dialogos": 80_000,
    },
}


def receta(nombre: str, arquitectura: str = "moderna") -> dict[str, Any]:
    """Devuelve una copia de la receta, con los interruptores de la arquitectura pedida."""
    if nombre not in RECETAS:
        raise ErrorDeConfig(f"No existe la receta «{nombre}». Las que hay: {', '.join(RECETAS)}.")
    if arquitectura not in ARQUITECTURAS:
        raise ErrorDeConfig(f"No existe la arquitectura «{arquitectura}». Las que hay: {', '.join(ARQUITECTURAS)}.")
    r = copy.deepcopy(RECETAS[nombre])
    r["receta"] = nombre
    r["arquitectura"] = arquitectura
    r["modelo"] = {**r["modelo"], **ARQUITECTURAS[arquitectura], "vocab": r["vocab"]}
    # Se validan aquí para que un error salga al crear el proyecto y no a mitad del entrenamiento.
    ConfigModelo.desde_dict(r["modelo"])
    ConfigEntreno.desde_dict(r["entreno"])
    ConfigEntreno.desde_dict(r["chat"])
    return r


def guardar_json(ruta: Path | str, valor: Any) -> None:
    """Escritura atómica: primero a un archivo temporal y luego se renombra. Si se va la luz
    a mitad, queda el archivo anterior entero y nunca uno a medias."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(json.dumps(valor, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta)


def leer_json(ruta: Path | str) -> Any:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def miles(numero: float) -> str:
    """Un entero con el punto de los miles, como se escribe en español: 1.234.567."""
    return f"{round(numero):,}".replace(",", ".")
