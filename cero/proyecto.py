"""Un proyecto = una carpeta con todo lo que hace falta para un modelo.

    proyectos/<nombre>/
        config.json          la receta (se puede editar a mano antes de cada etapa)
        tokenizador.json     el vocabulario aprendido
        datos/               el corpus convertido en números
        base/                el modelo «base»: solo sabe continuar texto      (cero entrenar)
        chat/                el modelo ajustado para conversar                 (cero afinar)

Dentro de `base/` y `chat/`:
        mejor.pt             el mejor modelo según la validación. Es autosuficiente: lleva
                             dentro los pesos, la forma del modelo y el tokenizador.
        ultimo.pt            todo lo necesario para reanudar un entrenamiento interrumpido
        metricas.jsonl       una línea por medición (lo que dibuja el laboratorio)
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import ConfigEntreno, ConfigModelo, ErrorDeConfig, guardar_json, leer_json, ORDEN
from .modelo import GPT
from .tokenizador import TokenizadorBPE

RAIZ = Path(__file__).resolve().parents[1]
ETAPAS = ("base", "chat")


def carpeta_proyectos() -> Path:
    return Path(os.environ.get("CERO_PROYECTOS") or RAIZ / "proyectos")


def carpeta_descargas() -> Path:
    return Path(os.environ.get("CERO_DESCARGAS") or RAIZ / "descargas")


class ErrorDeProyecto(RuntimeError):
    """Falta algo para poder continuar. El mensaje dice qué orden hay que ejecutar antes."""


class Proyecto:
    def __init__(self, nombre: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", nombre or ""):
            raise ErrorDeProyecto(
                f"«{nombre}» no vale como nombre de proyecto: usa letras, números, guiones o puntos (sin espacios)."
            )
        self.nombre = nombre
        self.carpeta = carpeta_proyectos() / nombre

    # ------------------------------------------------------------------ rutas
    @property
    def ruta_config(self) -> Path:
        return self.carpeta / "config.json"

    @property
    def ruta_tokenizador(self) -> Path:
        return self.carpeta / "tokenizador.json"

    @property
    def datos(self) -> Path:
        return self.carpeta / "datos"

    def etapa(self, nombre: str) -> Path:
        if nombre not in ETAPAS:
            raise ValueError(f"Etapa desconocida: {nombre!r} (hay {ETAPAS}).")
        return self.carpeta / nombre

    # ------------------------------------------------------------------ crear / abrir
    @classmethod
    def crear(cls, nombre: str, config: dict[str, Any], reemplazar: bool = False) -> "Proyecto":
        p = cls(nombre)
        if p.ruta_config.exists():
            if not reemplazar:
                raise ErrorDeProyecto(
                    f"Ya existe el proyecto «{nombre}». Elige otro nombre o añade --reemplazar para empezarlo de nuevo."
                )
            shutil.rmtree(p.carpeta)
        p.carpeta.mkdir(parents=True, exist_ok=True)
        guardar_json(p.ruta_config, config)
        activar(nombre)
        return p

    @classmethod
    def abrir(cls, nombre: str | None = None) -> "Proyecto":
        nombre = nombre or activo()
        if not nombre:
            raise ErrorDeProyecto(f"Todavía no hay ningún proyecto. Crea el primero con:  {ORDEN} todo --receta nano")
        p = cls(nombre)
        if not p.ruta_config.exists():
            existentes = ", ".join(listar()) or "ninguno"
            raise ErrorDeProyecto(f"No existe el proyecto «{nombre}». Proyectos disponibles: {existentes}.")
        return p

    # ------------------------------------------------------------------ contenido
    @property
    def config(self) -> dict[str, Any]:
        return leer_json(self.ruta_config)

    def guardar_config(self, config: dict[str, Any]) -> None:
        guardar_json(self.ruta_config, config)

    def config_modelo(self) -> ConfigModelo:
        return ConfigModelo.desde_dict(self.config["modelo"])

    def config_entreno(self, etapa: str) -> ConfigEntreno:
        return ConfigEntreno.desde_dict(self.config["entreno" if etapa == "base" else "chat"])

    def tokenizador(self) -> TokenizadorBPE:
        if not self.ruta_tokenizador.exists():
            raise ErrorDeProyecto(f"El proyecto «{self.nombre}» aún no tiene tokenizador. Ejecuta antes:  {ORDEN} datos")
        return TokenizadorBPE.cargar(self.ruta_tokenizador)

    def mejor(self, etapa: str) -> Path:
        return self.etapa(etapa) / "mejor.pt"

    def etapa_disponible(self, preferida: str = "auto") -> str:
        """Qué modelo usar: el de chat si existe, y si no el base."""
        if preferida in ETAPAS:
            if not self.mejor(preferida).exists():
                orden = f"{ORDEN} entrenar" if preferida == "base" else f"{ORDEN} afinar"
                raise ErrorDeProyecto(f"El proyecto «{self.nombre}» aún no tiene modelo «{preferida}». Ejecuta antes:  {orden}")
            return preferida
        for etapa in ("chat", "base"):
            if self.mejor(etapa).exists():
                return etapa
        raise ErrorDeProyecto(f"El proyecto «{self.nombre}» aún no tiene ningún modelo entrenado. Ejecuta antes:  {ORDEN} entrenar")

    def cargar(self, etapa: str = "auto", dispositivo: str = "cpu") -> tuple[GPT, TokenizadorBPE, dict[str, Any]]:
        """Carga un modelo entrenado, listo para generar. Devuelve (modelo, tokenizador, ficha)."""
        etapa = self.etapa_disponible(etapa)
        return cargar_modelo(self.mejor(etapa), dispositivo)

    def estado(self) -> dict[str, Any]:
        """Qué etapas están hechas. Lo usan `cero estado` y el laboratorio."""
        def ficha(etapa: str) -> dict[str, Any] | None:
            ruta = self.etapa(etapa) / "estado.json"
            return leer_json(ruta) if ruta.exists() else None

        manifiesto = self.datos / "manifiesto.json"
        chat = self.datos / "chat_manifiesto.json"
        config = self.config
        return {
            "nombre": self.nombre,
            "receta": config.get("receta"),
            "arquitectura": config.get("arquitectura"),
            "config": config,
            "datos": leer_json(manifiesto) if manifiesto.exists() else None,
            "dialogos": leer_json(chat) if chat.exists() else None,
            "base": ficha("base"),
            "chat": ficha("chat"),
            "base_listo": self.mejor("base").exists(),
            "chat_listo": self.mejor("chat").exists(),
        }


# ----------------------------------------------------------------------------------------
# Proyecto activo y listado
# ----------------------------------------------------------------------------------------
def activo() -> str | None:
    marca = carpeta_proyectos() / ".activo"
    if marca.exists():
        nombre = marca.read_text(encoding="utf-8").strip()
        if nombre and (carpeta_proyectos() / nombre / "config.json").exists():
            return nombre
    nombres = listar()
    return nombres[0] if len(nombres) == 1 else None


def activar(nombre: str) -> None:
    carpeta_proyectos().mkdir(parents=True, exist_ok=True)
    (carpeta_proyectos() / ".activo").write_text(nombre, encoding="utf-8")


def listar() -> list[str]:
    raiz = carpeta_proyectos()
    if not raiz.exists():
        return []
    return sorted(p.name for p in raiz.iterdir() if (p / "config.json").exists())


# ----------------------------------------------------------------------------------------
# Puntos de control
# ----------------------------------------------------------------------------------------
def guardar_punto(ruta: Path | str, contenido: dict[str, Any]) -> None:
    """Guarda de forma atómica: nunca queda un archivo a medias aunque se corte la corriente."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    try:
        torch.save(contenido, temporal)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def leer_punto(ruta: Path | str, dispositivo: str = "cpu") -> dict[str, Any]:
    """Lee un punto de control. `weights_only=True` solo acepta tensores y datos simples:
    un archivo .pt de origen dudoso no puede ejecutar código al abrirse."""
    return torch.load(Path(ruta), map_location=dispositivo, weights_only=True)


def cargar_modelo(ruta: Path | str, dispositivo: str = "cpu") -> tuple[GPT, TokenizadorBPE, dict[str, Any]]:
    """De un archivo `mejor.pt` a un modelo listo para usar, con su tokenizador."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorDeProyecto(f"No existe el modelo {ruta}.")
    punto = leer_punto(ruta, "cpu")
    try:
        cfg = ConfigModelo.desde_dict(punto["config_modelo"])
        tok = TokenizadorBPE.desde_dict(punto["tokenizador"])
    except (KeyError, ErrorDeConfig) as error:
        raise ErrorDeProyecto(f"{ruta} no es un modelo de este proyecto ({error}).") from error
    modelo = GPT(cfg)
    modelo.load_state_dict(punto["modelo"], strict=True)
    modelo.to(dispositivo).eval()
    ficha = {k: v for k, v in punto.items() if k not in ("modelo", "tokenizador")}
    ficha["ruta"] = str(ruta)
    return modelo, tok, ficha
