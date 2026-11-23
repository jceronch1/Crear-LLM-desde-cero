"""¿Dónde se hacen las cuentas? Elegir CPU o GPU y explicar qué se ha elegido.

El mismo código corre en los tres sitios; solo cambia la velocidad:

    cuda   tarjeta NVIDIA. Usa precisión mixta (bfloat16) si la tarjeta la admite.
    mps    chip Apple Silicon (M1, M2, ...).
    cpu    cualquier ordenador. Las recetas «nano» y «micro» están pensadas para esto.
"""

from __future__ import annotations

import contextlib
import os
import platform
from typing import Any

import torch


def elegir(pedido: str | None = "auto") -> str:
    """Convierte "auto" (o una petición concreta) en un dispositivo que existe en este equipo.

    Si se pide una GPU que no está disponible no se rompe nada: se avisa y se usa la CPU.
    La variable de entorno CERO_DISPOSITIVO (cpu o gpu) cambia lo que significa "auto": por
    ejemplo, `CERO_DISPOSITIVO=cpu` hace que todo el curso calcule en la CPU.
    """
    pedido = (pedido or "auto").lower()
    if pedido == "auto" and os.environ.get("CERO_DISPOSITIVO", "").lower() in ("cpu", "gpu", "cuda", "mps"):
        pedido = os.environ["CERO_DISPOSITIVO"].lower()
    if pedido in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if pedido in ("gpu", "mps", "auto") and _hay_mps():
        return "mps"
    if pedido in ("cpu", "auto"):
        return "cpu"
    if pedido in ("gpu", "cuda", "mps"):
        print(f"  (aviso) Se pidió «{pedido}» pero este equipo no lo tiene disponible: se usa la CPU.")
        return "cpu"
    raise ValueError(f"Dispositivo desconocido: {pedido!r}. Usa auto, cpu o gpu.")


def _hay_mps() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def preparar(dispositivo: str, hilos: int | None = None) -> None:
    """Ajustes de velocidad que no cambian el resultado de forma apreciable."""
    if hilos:
        torch.set_num_threads(int(hilos))
    if dispositivo == "cuda":
        # Permite a la GPU usar TF32 en las multiplicaciones de matrices: más rápido, misma calidad.
        torch.set_float32_matmul_precision("high")


def tipo_amp(dispositivo: str) -> torch.dtype | None:
    """Tipo numérico para la precisión mixta, o None si se calcula todo en float32.

    En GPU los pesos siguen en float32, pero las cuentas intermedias se hacen en bfloat16
    (16 bits): la mitad de memoria y bastante más velocidad. En CPU no compensa.
    """
    if dispositivo == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def autocast(dispositivo: str) -> contextlib.AbstractContextManager[Any]:
    """Contexto de precisión mixta: `with autocast(dispositivo): ...`. En CPU no hace nada."""
    tipo = tipo_amp(dispositivo)
    if tipo is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=tipo)


def sincronizar(dispositivo: str) -> None:
    """La GPU trabaja en diferido; para cronometrar hay que esperar a que termine."""
    if dispositivo == "cuda":
        torch.cuda.synchronize()


def describir(dispositivo: str) -> dict[str, Any]:
    """Los datos del equipo que importan para entrenar, listos para imprimir o enviar al laboratorio."""
    info: dict[str, Any] = {
        "dispositivo": dispositivo,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "procesador": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count() or 1,
        "hilos_pytorch": torch.get_num_threads(),
        "precision": "bfloat16 (mixta)" if tipo_amp(dispositivo) is not None else "float32",
    }
    if dispositivo == "cuda":
        propiedades = torch.cuda.get_device_properties(0)
        info["gpu"] = propiedades.name
        info["vram_gb"] = round(propiedades.total_memory / 2**30, 1)
        info["cuda"] = torch.version.cuda
    elif dispositivo == "mps":
        info["gpu"] = "Apple Silicon (MPS)"
    return info


def resumen(dispositivo: str) -> str:
    d = describir(dispositivo)
    if dispositivo == "cuda":
        return f"GPU {d['gpu']} ({d['vram_gb']} GB) · {d['precision']} · PyTorch {d['pytorch']}"
    if dispositivo == "mps":
        return f"GPU Apple Silicon · {d['precision']} · PyTorch {d['pytorch']}"
    return f"CPU con {d['hilos_pytorch']} hilos · {d['precision']} · PyTorch {d['pytorch']}"
