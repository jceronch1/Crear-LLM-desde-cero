"""Instalador de «LLM desde CERO». Solo usa la biblioteca estándar de Python.

    python instalar.py            detecta el equipo e instala lo que le corresponde
    python instalar.py --cpu      fuerza la versión para CPU (descarga pequeña, unos 130 MB)
    python instalar.py --gpu      exige GPU NVIDIA y avisa si no se consigue
    (otras opciones: --sin-curso, --reinstalar, --con-pip; ver `python instalar.py --help`)

Qué hace, en orden:
    1. comprueba la versión de Python (hace falta 3.11 o superior)
    2. crea un entorno aislado en la carpeta .venv (no toca el Python del sistema)
    3. instala PyTorch en la variante adecuada: CUDA si hay GPU NVIDIA, CPU si no
    4. COMPRUEBA que la GPU funciona de verdad con una multiplicación de matrices;
       si no funciona, prueba una versión anterior y, en último caso, instala la de CPU
    5. instala el proyecto, los cuadernos del curso y el lector del corpus
    6. ejecuta `cero comprobar`

Se puede ejecutar todas las veces que haga falta: lo que ya está instalado no se repite.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
PYTHON_VENV = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MINIMO = (3, 11)

# Variantes de PyTorch, de la más nueva a la más compatible. Comprobadas el 13-11-2026:
#   2.15.0 + CUDA 13.2   la versión estable actual. Necesita un controlador NVIDIA reciente
#                        y una GPU de la serie RTX 20 o posterior (CUDA 13 dejó fuera las GTX 10).
#   2.14.1 + CUDA 12.6   la última con CUDA 12: sirve para controladores y tarjetas más antiguos.
#   2.15.0 para CPU      funciona en cualquier equipo.
INDICE = "https://download.pytorch.org/whl/"
GPU_NUEVA = ("torch==2.15.0", INDICE + "cu132", "PyTorch 2.15.0 con CUDA 13.2 (unos 2,1 GB)")
GPU_ANTIGUA = ("torch==2.14.1", INDICE + "cu126", "PyTorch 2.14.1 con CUDA 12.6 (unos 2,5 GB)")
SOLO_CPU = ("torch==2.15.0", INDICE + "cpu", "PyTorch 2.15.0 para CPU (unos 130 MB)")
DE_PYPI = ("torch==2.15.0", None, "PyTorch 2.15.0")          # macOS: el paquete normal ya usa el chip de Apple

PRUEBA_GPU = (
    "import torch, sys\n"
    "ok = torch.cuda.is_available()\n"
    "if ok:\n"
    "    a = torch.randn(256, 256, device='cuda')\n"
    "    ok = bool(torch.isfinite((a @ a).sum()))\n"       # una operación real: detecta tarjetas no admitidas
    "sys.exit(0 if ok else 1)\n"
)


def decir(texto: str = "") -> None:
    print(texto, flush=True)


def paso(numero: int, texto: str) -> None:
    decir(f"\n[{numero}/5] {texto}")


def ejecutar(orden: list[str], silencioso: bool = False) -> bool:
    """Ejecuta una orden y dice si terminó bien."""
    try:
        return subprocess.run(orden, cwd=RAIZ, check=False, capture_output=silencioso).returncode == 0
    except OSError:
        return False


def salida(orden: list[str]) -> str:
    try:
        return subprocess.run(orden, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


USAR_PIP = False                 # --con-pip lo pone a True


def instalar(paquetes: list[str], indice: str | None = None) -> bool:
    """Instala paquetes en el entorno. Usa `uv` si está (mucho más rápido) y si no, `pip`."""
    if shutil.which("uv") and not USAR_PIP:
        orden = ["uv", "pip", "install", "--python", str(PYTHON_VENV), "--link-mode", "copy", *paquetes]
    else:
        orden = [str(PYTHON_VENV), "-m", "pip", "install", "--disable-pip-version-check", *paquetes]
    if indice:
        orden += ["--index-url", indice]
    return ejecutar(orden)


def gpu_nvidia() -> tuple[str, float] | None:
    """(nombre, versión del controlador) de la GPU NVIDIA, o None si no hay."""
    if not shutil.which("nvidia-smi"):
        return None
    linea = salida(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).splitlines()
    if not linea or "," not in linea[0]:
        return None
    nombre, controlador = (parte.strip() for parte in linea[0].split(",", 1))
    try:
        return nombre, float(".".join(controlador.split(".")[:2]))
    except ValueError:
        return nombre, 0.0


def torch_instalado() -> str:
    """Versión de PyTorch que ya hay en el entorno ("" si no hay)."""
    if not PYTHON_VENV.exists():
        return ""
    return salida([str(PYTHON_VENV), "-c", "import torch; print(torch.__version__)"])


def gpu_funciona() -> bool:
    return ejecutar([str(PYTHON_VENV), "-c", PRUEBA_GPU], silencioso=True)


def main() -> int:
    analizador = argparse.ArgumentParser(description="Instala «LLM desde CERO» en este equipo.")
    grupo = analizador.add_mutually_exclusive_group()
    grupo.add_argument("--cpu", action="store_true", help="instalar la versión para CPU aunque haya GPU")
    grupo.add_argument("--gpu", action="store_true", help="exigir GPU NVIDIA (falla si no se consigue)")
    analizador.add_argument("--sin-curso", action="store_true", help="no instalar Jupyter ni matplotlib")
    analizador.add_argument("--reinstalar", action="store_true", help="volver a instalar PyTorch aunque ya esté")
    analizador.add_argument("--con-pip", action="store_true", help="instalar con pip aunque exista uv")
    args = analizador.parse_args()
    global USAR_PIP
    USAR_PIP = args.con_pip
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    decir("══ LLM desde CERO · instalación ══")

    paso(1, "Comprobando Python")
    if sys.version_info < MINIMO:
        decir(f"  ✗ Este Python es el {platform.python_version()} y hace falta el {MINIMO[0]}.{MINIMO[1]} o superior.")
        decir("    Instala uno actual desde https://www.python.org/downloads/ y vuelve a ejecutar este instalador.")
        return 1
    decir(f"  ✔ Python {platform.python_version()} ({platform.system()} {platform.machine()})")

    paso(2, "Preparando el entorno aislado (.venv)")
    if not PYTHON_VENV.exists():
        import venv

        venv.EnvBuilder(with_pip=True, clear=VENV.exists()).create(VENV)
        decir("  ✔ Entorno creado")
    else:
        if not ejecutar([str(PYTHON_VENV), "-m", "pip", "--version"], silencioso=True) and not shutil.which("uv"):
            ejecutar([str(PYTHON_VENV), "-m", "ensurepip", "--upgrade"], silencioso=True)
        decir("  ✔ El entorno ya existía")

    paso(3, "Instalando PyTorch")
    gpu = None if args.cpu else gpu_nvidia()
    if platform.system() == "Darwin":
        planes = [DE_PYPI]
        decir("  Equipo Apple: se instala el PyTorch normal (usa el chip M1/M2/M3 si lo hay).")
    elif gpu is None:
        planes = [SOLO_CPU]
        if args.gpu:
            decir("  ✗ Se pidió --gpu pero no se encuentra una GPU NVIDIA (falta `nvidia-smi`). Instala el controlador de NVIDIA.")
            return 1
        decir("  " + ("Se pidió la versión para CPU." if args.cpu else "No se detecta GPU NVIDIA: se instala la versión para CPU."))
    else:
        nombre, controlador = gpu
        decir(f"  GPU detectada: {nombre} (controlador {controlador:g})")
        # El número del controlador es solo una pista para no descargar 2 GB en vano; lo que
        # decide de verdad es la comprobación que se hace después de instalar.
        planes = ([GPU_NUEVA] if controlador >= 580 else []) + [GPU_ANTIGUA] + ([] if args.gpu else [SOLO_CPU])

    actual = torch_instalado()
    quiere_gpu = gpu is not None and platform.system() != "Darwin"
    # Si ya hay un PyTorch que sirve, no se toca. (Una versión con CUDA también funciona en CPU.)
    ya_vale = bool(actual) and not args.reinstalar and (gpu_funciona() if quiere_gpu else True)
    if ya_vale:
        decir(f"  ✔ PyTorch {actual} ya está instalado y funciona" + (" con la GPU" if quiere_gpu else ""))
    else:
        instalado = False
        for paquete, indice, descripcion in planes:
            decir(f"  Descargando {descripcion}. Puede tardar varios minutos...")
            if not instalar([paquete], indice):
                decir("  (no se pudo descargar; se prueba la siguiente opción)")
                continue
            if indice in (GPU_NUEVA[1], GPU_ANTIGUA[1]) and not gpu_funciona():
                decir("  PyTorch se instaló, pero la GPU no responde con esta versión. Se prueba la siguiente opción.")
                continue
            instalado = True
            break
        if not instalado:
            decir("  ✗ No se pudo instalar PyTorch. Revisa la conexión a internet y vuelve a ejecutar el instalador.")
            if args.gpu:
                decir("    Con --gpu no se instala la versión para CPU. Actualiza el controlador de NVIDIA o quita --gpu.")
            return 1
        decir(f"  ✔ PyTorch {torch_instalado()} instalado")
        if quiere_gpu and not gpu_funciona():
            decir("  (aviso) Se ha instalado la versión para CPU: todo funciona, solo que más despacio.")
            decir("          Actualizar el controlador de NVIDIA y repetir el instalador suele arreglarlo.")

    paso(4, "Instalando el proyecto" + ("" if args.sin_curso else ", los cuadernos del curso") + " y el lector del corpus")
    extras = "corpus,dev" if args.sin_curso else "todo"
    if not instalar(["-e", f".[{extras}]"]):
        decir("  ✗ Falló la instalación del proyecto. Mira el mensaje de arriba, corrige y repite.")
        return 1
    decir("  ✔ Proyecto instalado")

    paso(5, "Comprobación final")
    if not ejecutar([str(PYTHON_VENV), "-X", "utf8", "-m", "cero", "comprobar", "--rapido"]):
        decir("  ✗ La comprobación final falló. Copia el mensaje de arriba y pide ayuda a tu profesor.")
        return 1

    orden = r".\cero" if os.name == "nt" else "./cero.sh"
    decir("\n══ Instalación terminada ══")
    if os.name == "nt":
        decir("  Lo más fácil: doble clic en  menu.cmd  (un menú con todas las opciones).")
        decir("  O desde una terminal abierta en esta carpeta:")
    else:
        decir("  Primeros pasos, desde esta carpeta:")
    decir(f"    {orden} comprobar              cuánto tardará cada receta en este equipo")
    decir(f"    {orden} todo --receta nano     tu primer LLM con su chatbot, en unos minutos")
    decir(f"    {orden} chat                   conversar con él")
    decir(f"    {orden} lab                    el laboratorio: verlo por dentro, en el navegador")
    decir(f"    {orden} curso                  las clases del curso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
