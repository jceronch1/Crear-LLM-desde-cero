"""La herramienta de línea de órdenes: `cero <orden>`.

El recorrido completo, paso a paso (cada orden es una etapa y una clase del curso):

    cero comprobar                 ¿está todo instalado? ¿cuánto tardará en este equipo?
    cero nuevo mi-llm              crea un proyecto a partir de una receta
    cero datos                     corpus -> tokenizador -> números
    cero entrenar                  preentrena el modelo base
    cero generar "Había una vez"   el modelo base continúa un texto
    cero afinar                    lo ajusta para conversar
    cero chat                      conversación en la consola
    cero evaluar                   mide qué aprendió y qué no

O todo seguido:   cero todo --receta nano      Y con interfaz:   cero lab
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import ARQUITECTURAS, ConfigModelo, ErrorDeConfig, miles, ORDEN, receta, RECETAS
from .proyecto import RAIZ, ErrorDeProyecto, Proyecto, activar, activo, listar


# ----------------------------------------------------------------------------------------
# Utilidades de consola
# ----------------------------------------------------------------------------------------
def _utf8() -> None:
    """Las consolas de Windows a veces no están en UTF-8; así los acentos nunca rompen el programa."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def titulo(texto: str) -> None:
    print(f"\n── {texto} " + "─" * max(3, 76 - len(texto)))


def _progreso():
    """Una línea que se reescribe sola mientras se entrena (solo si hay una consola de verdad)."""
    if not sys.stdout.isatty():
        return None

    def mostrar(paso: int, pasos: int, perdida: float, velocidad: float) -> None:
        hecho = int(24 * paso / pasos)
        barra = "█" * hecho + "░" * (24 - hecho)
        print(f"\r  {barra} {paso:>6}/{pasos} · pérdida {perdida:6.3f} · {miles(velocidad)} tok/s   ", end="", flush=True)
        if paso == pasos:
            print("\r" + " " * 78 + "\r", end="")

    return mostrar


def _avisar(mensaje: str) -> None:
    if sys.stdout.isatty():
        print("\r" + " " * 78 + "\r", end="")                # borrar la barra de progreso antes de escribir
    print(mensaje, flush=True)


def _dispositivo(args: argparse.Namespace) -> str:
    from . import dispositivo as disp

    return disp.elegir(getattr(args, "dispositivo", "auto"))


def _abrir(args: argparse.Namespace) -> Proyecto:
    return Proyecto.abrir(getattr(args, "proyecto", None))


def _senal_de_pausa(proyecto: Proyecto):
    """Una forma de pausar desde fuera (la usa el laboratorio): crear el archivo PARAR en el proyecto.

    El entrenamiento lo mira después de cada paso; si existe, guarda y se detiene con orden.
    """
    marca = proyecto.carpeta / "PARAR"
    marca.unlink(missing_ok=True)

    def debe_parar() -> bool:
        if marca.exists():
            marca.unlink(missing_ok=True)
            return True
        return False

    return debe_parar


def _ajustar_pasos(proyecto: Proyecto, etapa: str, args: argparse.Namespace, dispositivo: str) -> None:
    """Aplica --pasos o --minutos a la configuración del proyecto (y lo deja guardado)."""
    pasos = getattr(args, "pasos", None)
    minutos = getattr(args, "minutos", None)
    if not pasos and not minutos:
        return
    config = proyecto.config
    clave = "entreno" if etapa == "base" else "chat"
    if minutos:
        from .entrenar import medir_velocidad

        print(f"  Midiendo la velocidad de este equipo para ajustar el entrenamiento a {minutos:g} minutos...")
        cfg = ConfigModelo.desde_dict(config["modelo"])
        velocidad = medir_velocidad(cfg, int(config[clave]["lote"]), dispositivo, segundos=2.5)
        pasos = max(40, int(minutos * 60 * 0.85 * velocidad["pasos_s"]))   # 15 % de margen para las evaluaciones
        print(f"  {velocidad['pasos_s']:.1f} pasos por segundo → {miles(pasos)} pasos.")
    config[clave]["pasos"] = int(pasos)
    config[clave]["evaluar_cada"] = max(10, min(int(config[clave]["evaluar_cada"]), int(pasos) // 5))
    config[clave]["calentamiento"] = max(5, min(int(config[clave]["calentamiento"]), int(pasos) // 10))
    proyecto.guardar_config(config)


# ----------------------------------------------------------------------------------------
# Órdenes
# ----------------------------------------------------------------------------------------
def cmd_comprobar(args: argparse.Namespace) -> None:
    import torch

    from . import dispositivo as disp
    from .entrenar import medir_velocidad
    from .modelo import GPT

    titulo("Comprobación del equipo")
    elegido = disp.elegir("auto")
    d = disp.describir(elegido)
    print(f"  cero {__version__} · Python {d['python']} · PyTorch {d['pytorch']} · {d['sistema']}")
    print(f"  Procesador: {d['nucleos_logicos']} núcleos lógicos, PyTorch usará {d['hilos_pytorch']} hilos")
    if elegido == "cuda":
        print(f"  GPU: {d['gpu']} ({d['vram_gb']} GB) · CUDA {d['cuda']} · precisión {d['precision']}")
    elif elegido == "mps":
        print("  GPU: Apple Silicon (MPS)")
    else:
        print("  GPU: no se detecta ninguna compatible. No pasa nada: las recetas «nano» y «micro» van bien en CPU.")

    # Una comprobación real del modelo: la atención «a mano» y la de PyTorch deben coincidir.
    torch.manual_seed(0)
    cfg = ConfigModelo(vocab=300, contexto=32, d_modelo=64, n_capas=2, n_cabezas=4)
    modelo = GPT(cfg).eval()
    x = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        diferencia = float((modelo(x)[0] - modelo.mirar(x)[0]).abs().max())
    print(f"  Prueba del modelo: {'correcta' if diferencia < 1e-4 else 'FALLA'} (atención a mano frente a PyTorch: diferencia {diferencia:.1e})")
    try:
        import pyarrow  # noqa: F401

        print("  Corpus real: pyarrow instalado (recetas micro, mini y base disponibles)")
    except ImportError:
        print("  Corpus real: falta pyarrow (solo hace falta para las recetas micro, mini y base)")

    if args.rapido:
        return
    titulo("¿Cuánto tardará cada receta en este equipo?")
    print("  (se entrena unos segundos con datos al azar para medirlo)\n")
    dispositivos = ["cpu"] + ([elegido] if elegido != "cpu" else [])
    print(f"  {'receta':<7} {'parámetros':>11}   " + "   ".join(f"{('CPU' if x == 'cpu' else 'GPU'):<27}" for x in dispositivos))
    referencia_cpu: tuple[float, float] | None = None       # (tokens/s, coste por token) de la última receta medida en CPU
    hay_estimados = False
    for nombre in RECETAS:
        r = receta(nombre)
        cfg = ConfigModelo.desde_dict(r["modelo"])
        modelo = GPT(cfg)
        n = modelo.n_parametros()
        coste = n                                            # las cuentas por token crecen, a ojo, como los parámetros
        del modelo
        tokens_base = r["entreno"]["pasos"] * r["entreno"]["lote"] * cfg.contexto
        tokens_chat = r["chat"]["pasos"] * r["chat"]["lote"] * cfg.contexto * 0.55
        celdas = []
        for dispositivo in dispositivos:
            marca = " "
            if dispositivo == "cpu" and nombre in ("mini", "base") and referencia_cpu:
                velocidad = referencia_cpu[0] * referencia_cpu[1] / coste          # extrapolado: no se mide para no esperar minutos
                marca, hay_estimados = "*", True
            else:
                disp.preparar(dispositivo)
                velocidad = medir_velocidad(cfg, min(r["entreno"]["lote"], 32), dispositivo, segundos=1.5)["tokens_s"]
                if dispositivo == "cpu":
                    referencia_cpu = (velocidad, coste)
            segundos = (tokens_base + tokens_chat) / velocidad
            celdas.append(f"{miles(velocidad):>8} tok/s ≈ {_duracion(segundos) + marca:<9}")
        print(f"  {nombre:<7} {_coma(n / 1e6):>9} M   " + "   ".join(f"{c:<27}" for c in celdas))
    print("\n  Tiempo total aproximado de «cero todo» (preentrenar + ajustar para chat), sin contar la descarga del corpus.")
    if hay_estimados:
        print("  * Estimado a partir de la receta micro (medirlo de verdad llevaría varios minutos).")
    print(f"  Para ceñirte al tiempo de una clase:  {ORDEN} todo --receta nano --minutos 5")


def _coma(numero: float, decimales: int = 1) -> str:
    """Un número con coma decimal, como se escribe en español: 0,9."""
    return f"{numero:.{decimales}f}".replace(".", ",")


def _duracion(segundos: float) -> str:
    if segundos < 90:
        return f"{segundos:.0f} s"
    if segundos < 5400:
        return f"{segundos / 60:.0f} min"
    return f"{_coma(segundos / 3600)} h"


def cmd_recetas(args: argparse.Namespace) -> None:
    from .modelo import GPT

    titulo("Recetas disponibles")
    for nombre in RECETAS:
        r = receta(nombre)
        cfg = ConfigModelo.desde_dict(r["modelo"])
        n = GPT(cfg).n_parametros()
        print(f"\n  {nombre}  ·  {_coma(n / 1e6)} M de parámetros")
        print(f"      {r['descripcion']}")
        print(f"      {cfg.n_capas} capas · {cfg.n_cabezas} cabezas · ancho {cfg.d_modelo} · contexto {cfg.contexto} · vocabulario {miles(r['vocab'])}")
    print("\n  Arquitecturas (--arquitectura): " + ", ".join(ARQUITECTURAS) + "   [por defecto: moderna]")
    print(f"  Crear un proyecto:  {ORDEN} nuevo mi-llm --receta nano")


def cmd_nuevo(args: argparse.Namespace) -> Proyecto:
    config = receta(args.receta, args.arquitectura)
    if getattr(args, "texto", None):
        config["corpus"] = {"fuente": "texto", "ruta": str(Path(args.texto).expanduser().resolve())}
    proyecto = Proyecto.crear(args.nombre, config, reemplazar=args.reemplazar)
    print(f"  Proyecto «{proyecto.nombre}» creado con la receta «{args.receta}» y la arquitectura «{args.arquitectura}».")
    print(f"  Su configuración se puede editar aquí: {proyecto.ruta_config}")
    return proyecto


def cmd_datos(args: argparse.Namespace) -> None:
    from . import datos

    proyecto = _abrir(args)
    titulo(f"Datos y tokenizador · proyecto «{proyecto.nombre}»")
    datos.preparar(proyecto, _avisar)


def cmd_entrenar(args: argparse.Namespace) -> None:
    from .entrenar import entrenar_base

    proyecto = _abrir(args)
    dispositivo = _dispositivo(args)
    titulo(f"Preentrenamiento · proyecto «{proyecto.nombre}»")
    if not (proyecto.datos / "entreno.bin").exists():
        from . import datos

        datos.preparar(proyecto, _avisar)
    if not args.reanudar:
        _ajustar_pasos(proyecto, "base", args, dispositivo)
    resumen = entrenar_base(proyecto, dispositivo, reanudar=args.reanudar, avisar=_avisar, progreso=_progreso(),
                            debe_parar=_senal_de_pausa(proyecto))
    _cierre(resumen, f"{ORDEN} generar \"Había una vez\"      después:  {ORDEN} afinar")


def cmd_afinar(args: argparse.Namespace) -> None:
    from .chat import afinar

    proyecto = _abrir(args)
    dispositivo = _dispositivo(args)
    titulo(f"Ajuste para conversar (SFT) · proyecto «{proyecto.nombre}»")
    if not args.reanudar:
        _ajustar_pasos(proyecto, "chat", args, dispositivo)
    resumen = afinar(proyecto, dispositivo, reanudar=args.reanudar, avisar=_avisar, progreso=_progreso(),
                     debe_parar=_senal_de_pausa(proyecto))
    _cierre(resumen, f"{ORDEN} chat      y para medirlo:  {ORDEN} evaluar")


def _cierre(resumen: dict[str, Any], siguiente: str) -> None:
    if resumen["estado"] == "terminado" and resumen.get("perdida_val") is not None:
        print(f"\n  Terminado en {_duracion(resumen['segundos'])}: pérdida de validación {resumen['perdida_val']:.3f} "
              f"(perplejidad {resumen['perplejidad']}) · {miles(resumen['tokens_s'])} tokens por segundo.")
        print(f"  Siguiente paso:  {siguiente}")


def cmd_generar(args: argparse.Namespace) -> None:
    from .generar import generar

    proyecto = _abrir(args)
    etapa = proyecto.etapa_disponible(args.etapa or ("base" if proyecto.mejor("base").exists() else "auto"))
    modelo, tok, ficha = proyecto.cargar(etapa, _dispositivo(args))
    texto = args.texto or proyecto.config.get("muestra", "Había una vez")
    print(f"  [modelo «{etapa}» de «{proyecto.nombre}» · paso {ficha['paso']} · temperatura {args.temperatura}]\n")
    ids = [tok.id("<|inicio|>")] + tok.codificar_texto(texto)
    for i in range(args.muestras):
        print(texto, end="", flush=True)
        flujo = tok.flujo()
        semilla = None if args.semilla is None else args.semilla + i
        for token in generar(modelo, ids, args.tokens, args.temperatura, args.top_k, args.top_p,
                             parar=tok.especiales.values(), semilla=semilla):
            print(flujo.agregar(token), end="", flush=True)
        print(flujo.terminar() + "\n")


def cmd_chat(args: argparse.Namespace) -> None:
    from .chat import NOMBRE, a, codificar_dialogo, preparar_entrada, responder, u

    proyecto = _abrir(args)
    etapa = proyecto.etapa_disponible(args.etapa or "auto")
    modelo, tok, ficha = proyecto.cargar(etapa, _dispositivo(args))
    titulo(f"Chat con «{proyecto.nombre}» · modelo «{etapa}» · {miles(ficha['parametros'])} parámetros")
    if etapa == "base":
        print("  (aviso) Este modelo aún no está ajustado para conversar: continuará tu texto como si fuera un cuento.")
        print(f"          Ajústalo con:  {ORDEN} afinar")
    print("  Escribe y pulsa Intro.   /nuevo borra la conversación · /dentro muestra lo que ve el modelo · /salir termina\n")
    mensajes: list[dict[str, str]] = []
    while True:
        try:
            entrada = input("tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entrada:
            continue
        if entrada in ("/salir", "/exit", "/quit"):
            break
        if entrada == "/nuevo":
            mensajes = []
            print("  (conversación nueva)\n")
            continue
        if entrada == "/dentro":
            ids, _ = codificar_dialogo(tok, mensajes, abrir_respuesta=True)
            print(f"  El modelo ve {len(ids)} de {modelo.cfg.contexto} tokens:")
            print("  " + " ".join(f"[{tok.texto_de(i)}]" for i in ids).replace("\n", "↵") + "\n")
            continue
        mensajes.append(u(entrada))
        _, olvidados = preparar_entrada(tok, mensajes, modelo.cfg.contexto, modelo.cfg.contexto // 3)
        if olvidados:
            print(f"  (la conversación no cabe en {modelo.cfg.contexto} tokens: el modelo ya no ve los {olvidados} mensajes más antiguos)")
        print(f"{NOMBRE.lower()} > ", end="", flush=True)
        texto = ""
        for trozo in responder(modelo, tok, mensajes, args.tokens, args.temperatura, args.top_k, args.top_p):
            print(trozo, end="", flush=True)
            texto += trozo
        print("\n")
        mensajes.append(a(texto.strip()))


def cmd_evaluar(args: argparse.Namespace) -> None:
    from . import dispositivo as disp
    from .chat import evaluar
    from .config import guardar_json
    from .datos import Lotes
    from .entrenar import medir_perdida

    proyecto = _abrir(args)
    dispositivo = _dispositivo(args)
    disp.preparar(dispositivo)
    config = proyecto.config
    titulo(f"Evaluación · proyecto «{proyecto.nombre}»")
    informe: dict[str, Any] = {}
    if proyecto.mejor("base").exists():
        modelo, tok, ficha = proyecto.cargar("base", dispositivo)
        lotes = Lotes(proyecto.datos / "validacion.bin", modelo.cfg.contexto, 32, 4321, dispositivo)
        perdida = medir_perdida(modelo, lotes, 40, dispositivo)
        cpt = (proyecto.estado()["datos"] or {}).get("caracteres_por_token", 0)
        informe["base"] = {"perdida_val": round(perdida, 4), "perplejidad": round(math.exp(perdida), 3),
                           "bits_por_caracter": round(perdida / math.log(2) / cpt, 3) if cpt else None}
        print(f"  Modelo base (paso {ficha['paso']}):")
        print(f"    pérdida de validación   {perdida:.3f}   (azar puro: {math.log(modelo.cfg.vocab):.3f})")
        print(f"    perplejidad             {math.exp(perdida):.2f}   (entre cuántos tokens «duda» de media)")
        if cpt:
            print(f"    bits por carácter       {informe['base']['bits_por_caracter']:.3f}   (comparable entre tokenizadores distintos)")
    if proyecto.mejor("chat").exists():
        modelo, tok, ficha = proyecto.cargar("chat", dispositivo)
        print(f"\n  Chatbot (paso {ficha['paso']}): preguntas nuevas con respuesta comprobable, generando sin azar\n")
        inicio = time.time()
        informe["chat"] = evaluar(modelo, tok, config["corpus"]["fuente"], por_prueba=args.casos, avisar=print)
        total = sum(r["total"] for r in informe["chat"].values())
        paradas = sum(r["se_detiene"] for r in informe["chat"].values())
        print(f"\n  Se detiene solo (escribe <|fin|>) en {paradas}/{total} respuestas · {time.time() - inicio:.0f} s")
        print("  Un ejemplo de cada prueba queda guardado en evaluacion.json")
    if not informe:
        raise ErrorDeProyecto(f"El proyecto «{proyecto.nombre}» aún no tiene nada que evaluar. Ejecuta antes:  {ORDEN} entrenar")
    guardar_json(proyecto.carpeta / "evaluacion.json", informe)


def cmd_estado(args: argparse.Namespace) -> None:
    proyecto = _abrir(args)
    e = proyecto.estado()
    titulo(f"Proyecto «{proyecto.nombre}» · receta {e['receta']} · arquitectura {e['arquitectura']}")
    m = e["config"]["modelo"]
    print(f"  Modelo: {m['n_capas']} capas · {m['n_cabezas']} cabezas · ancho {m['d_modelo']} · contexto {m['contexto']} · vocabulario {miles(m['vocab'])}")

    def linea(hecho: bool, nombre: str, detalle: str, orden: str) -> None:
        print(f"  {'✔' if hecho else '·'} {nombre:<22} {detalle if hecho else 'pendiente  →  ' + orden}")

    d = e["datos"]
    linea(bool(d), "1. Datos y tokenizador", d and f"{miles(d['partes']['entreno']['tokens'])} tokens · {d['caracteres_por_token']} caracteres por token", f"{ORDEN} datos")
    for etapa, nombre, orden in (("base", "2. Modelo base", f"{ORDEN} entrenar"), ("chat", "3. Chatbot", f"{ORDEN} afinar")):
        f = e[etapa]
        detalle = ""
        if f:
            detalle = f"{f['estado']} · paso {miles(f['paso'])}/{miles(f['pasos'])} · validación {f['perdida_val']} · {_duracion(f['segundos'])} en {f['dispositivo'].split(' · ')[0]}"
        linea(bool(f) and e[f"{etapa}_listo"], nombre, detalle, orden)
    print(f"\n  Carpeta: {proyecto.carpeta}")


def cmd_proyectos(args: argparse.Namespace) -> None:
    nombres = listar()
    if not nombres:
        print(f"  Todavía no hay proyectos. Crea el primero con:  {ORDEN} todo --receta nano")
        return
    actual = activo()
    titulo("Proyectos")
    for nombre in nombres:
        e = Proyecto(nombre).estado()
        etapas = "chatbot listo" if e["chat_listo"] else "modelo base listo" if e["base_listo"] else "datos listos" if e["datos"] else "recién creado"
        print(f"  {'►' if nombre == actual else ' '} {nombre:<24} receta {e['receta']:<6} {etapas}")
    print(f"\n  El marcado con ► es el proyecto activo. Para cambiarlo:  {ORDEN} usar NOMBRE")


def cmd_usar(args: argparse.Namespace) -> None:
    proyecto = Proyecto.abrir(args.nombre)
    activar(proyecto.nombre)
    print(f"  Proyecto activo: «{proyecto.nombre}».")


def cmd_tokenizar(args: argparse.Namespace) -> None:
    proyecto = _abrir(args)
    tok = proyecto.tokenizador()
    piezas = tok.piezas(args.texto)
    print(f"  {len(args.texto)} caracteres → {len(piezas)} tokens ({len(args.texto) / max(len(piezas), 1):.2f} caracteres por token)\n")
    print("  " + " ".join(f"[{p['texto']}]" for p in piezas).replace("\n", "↵"))
    print("\n  " + " ".join(str(p["id"]) for p in piezas))


def cmd_todo(args: argparse.Namespace) -> None:
    from . import datos
    from .chat import afinar
    from .entrenar import entrenar_base

    inicio = time.time()
    dispositivo = _dispositivo(args)
    if not args.nombre:                                      # nombre por defecto: el de la receta (y la arquitectura, si no es la moderna)
        args.nombre = args.receta if args.arquitectura == "moderna" else f"{args.receta}-{args.arquitectura}"
    titulo(f"Un LLM desde cero · receta «{args.receta}»")
    proyecto = cmd_nuevo(args)
    titulo("1/3 · Datos y tokenizador")
    datos.preparar(proyecto, _avisar)
    if args.minutos:
        reparto = argparse.Namespace(pasos=None, minutos=args.minutos * 0.7)
        _ajustar_pasos(proyecto, "base", reparto, dispositivo)
        reparto.minutos = args.minutos * 0.3
        _ajustar_pasos(proyecto, "chat", reparto, dispositivo)
    titulo("2/3 · Preentrenamiento: aprender a continuar texto")
    pausa = _senal_de_pausa(proyecto)
    base = entrenar_base(proyecto, dispositivo, avisar=_avisar, progreso=_progreso(), debe_parar=pausa)
    if base["estado"] != "terminado":
        print(f"  Para continuar donde se quedó:  {ORDEN} entrenar --reanudar")
        return
    titulo("3/3 · Ajuste fino: aprender a conversar")
    chat = afinar(proyecto, dispositivo, avisar=_avisar, progreso=_progreso(), debe_parar=pausa)
    if chat["estado"] != "terminado":
        print(f"  Para continuar donde se quedó:  {ORDEN} afinar --reanudar")
        return
    titulo("Listo")
    print(f"  Tu modelo de {miles(base['parametros'])} parámetros está entrenado ({_duracion(time.time() - inicio)} en total).")
    print(f"  Habla con él:        {ORDEN} chat")
    print(f"  Míralo por dentro:   {ORDEN} lab")
    print(f"  Mide qué aprendió:   {ORDEN} evaluar")


def cmd_lab(args: argparse.Namespace) -> None:
    from .lab.servidor import servir

    servir(puerto=args.puerto, abrir=not args.no_abrir, dispositivo=args.dispositivo)


def cmd_curso(args: argparse.Namespace) -> None:
    carpeta = RAIZ / "curso"
    try:
        import jupyterlab  # noqa: F401
    except ImportError:
        print("  Los cuadernos del curso están en:", carpeta)
        print("  Para abrirlos hace falta Jupyter:  python instalar.py   (o: pip install jupyterlab)")
        print("  También se abren directamente con VS Code.")
        return
    print(f"  Abriendo los cuadernos del curso ({carpeta}). Ctrl+C aquí para cerrar Jupyter.")
    subprocess.run([sys.executable, "-m", "jupyterlab", f"--notebook-dir={carpeta}"], check=False)


# ----------------------------------------------------------------------------------------
# Analizador de argumentos
# ----------------------------------------------------------------------------------------
def crear_analizador() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cero", description="Un LLM y su chatbot desde cero, paso a paso.",
                                epilog=f"Primer uso:  {ORDEN} comprobar   y después   {ORDEN} todo --receta nano")
    p.add_argument("--version", action="version", version=f"cero {__version__}")
    sub = p.add_subparsers(dest="orden", metavar="ORDEN")

    def orden(nombre: str, funcion, ayuda: str) -> argparse.ArgumentParser:
        s = sub.add_parser(nombre, help=ayuda, description=ayuda)
        s.set_defaults(funcion=funcion)
        return s

    def con_proyecto(s: argparse.ArgumentParser) -> None:
        s.add_argument("-p", "--proyecto", help="nombre del proyecto (por defecto, el activo)")

    def con_dispositivo(s: argparse.ArgumentParser) -> None:
        s.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "gpu"], help="dónde calcular (por defecto: la GPU si hay)")

    def con_receta(s: argparse.ArgumentParser) -> None:
        s.add_argument("--receta", default="nano", choices=list(RECETAS), help="tamaño del proyecto (por defecto: nano)")
        s.add_argument("--arquitectura", default="moderna", choices=list(ARQUITECTURAS), help="generación del Transformer (por defecto: moderna)")
        s.add_argument("--texto", help="entrenar con tus propios textos: un archivo .txt o una carpeta")
        s.add_argument("--reemplazar", action="store_true", help="si el proyecto ya existe, empezarlo de nuevo")

    def con_entreno(s: argparse.ArgumentParser) -> None:
        s.add_argument("--pasos", type=int, help="número de pasos de entrenamiento (cambia el de la receta)")
        s.add_argument("--minutos", type=float, help="ajustar los pasos para que tarde más o menos estos minutos")
        s.add_argument("--reanudar", action="store_true", help="continuar un entrenamiento pausado")

    def con_mandos(s: argparse.ArgumentParser, tokens: int, temperatura: float) -> None:
        s.add_argument("--etapa", choices=["base", "chat"], help="qué modelo usar")
        s.add_argument("--tokens", type=int, default=tokens, help=f"máximo de tokens nuevos (por defecto {tokens})")
        s.add_argument("--temperatura", type=float, default=temperatura, help=f"0 = siempre lo más probable (por defecto {temperatura})")
        s.add_argument("--top-k", dest="top_k", type=int, default=40, help="quedarse con los k tokens más probables (por defecto 40)")
        s.add_argument("--top-p", dest="top_p", type=float, default=None, help="muestreo de núcleo, p. ej. 0.9")

    s = orden("comprobar", cmd_comprobar, "Comprueba la instalación y estima cuánto tardará cada receta aquí.")
    s.add_argument("--rapido", action="store_true", help="no medir velocidades")

    orden("recetas", cmd_recetas, "Muestra las recetas y arquitecturas disponibles.")

    s = orden("todo", cmd_todo, "Hace el recorrido completo: datos, preentrenamiento y ajuste para chat.")
    s.add_argument("nombre", nargs="?", help="nombre del proyecto (por defecto, el de la receta)")
    con_receta(s)
    con_dispositivo(s)
    s.add_argument("--minutos", type=float, help="tiempo total aproximado de entrenamiento")

    s = orden("nuevo", cmd_nuevo, "Crea un proyecto a partir de una receta.")
    s.add_argument("nombre", help="nombre del proyecto, sin espacios")
    con_receta(s)

    s = orden("datos", cmd_datos, "Etapa 1: lee el corpus, entrena el tokenizador y lo convierte todo en números.")
    con_proyecto(s)

    s = orden("entrenar", cmd_entrenar, "Etapa 2: preentrena el modelo base (aprende a continuar texto).")
    con_proyecto(s), con_dispositivo(s), con_entreno(s)

    s = orden("generar", cmd_generar, "Continúa un texto con el modelo.")
    s.add_argument("texto", nargs="?", help="comienzo del texto (por defecto: «Había una vez»)")
    con_proyecto(s), con_dispositivo(s), con_mandos(s, 120, 0.8)
    s.add_argument("--semilla", type=int, help="fijar el azar para repetir el resultado")
    s.add_argument("--muestras", type=int, default=1, help="cuántas continuaciones escribir")

    s = orden("afinar", cmd_afinar, "Etapa 3: ajuste fino para conversar (SFT).")
    con_proyecto(s), con_dispositivo(s), con_entreno(s)

    s = orden("chat", cmd_chat, "Conversa con tu modelo en la consola.")
    con_proyecto(s), con_dispositivo(s), con_mandos(s, 200, 0.7)

    s = orden("evaluar", cmd_evaluar, "Mide la pérdida del modelo base y las habilidades del chatbot.")
    con_proyecto(s), con_dispositivo(s)
    s.add_argument("--casos", type=int, default=25, help="preguntas por prueba (por defecto 25)")

    s = orden("tokenizar", cmd_tokenizar, "Muestra cómo trocea un texto el tokenizador del proyecto.")
    s.add_argument("texto")
    con_proyecto(s)

    s = orden("estado", cmd_estado, "Muestra qué etapas del proyecto están hechas.")
    con_proyecto(s)

    orden("proyectos", cmd_proyectos, "Lista los proyectos.")

    s = orden("usar", cmd_usar, "Cambia el proyecto activo.")
    s.add_argument("nombre")

    s = orden("lab", cmd_lab, "Abre el laboratorio: la interfaz web para ver el modelo por dentro y conversar.")
    s.add_argument("--puerto", type=int, default=8765)
    s.add_argument("--no-abrir", action="store_true", help="no abrir el navegador automáticamente")
    s.add_argument("--dispositivo", default="cpu", choices=["auto", "cpu", "gpu"],
                   help="dónde ejecutar los modelos del laboratorio (por defecto la CPU, para dejar la GPU libre al entrenar)")

    orden("curso", cmd_curso, "Abre los cuadernos del curso en Jupyter.")
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8()
    analizador = crear_analizador()
    args = analizador.parse_args(argv)
    if not getattr(args, "funcion", None):
        analizador.print_help()
        return 0
    try:
        args.funcion(args)
    except (ErrorDeProyecto, ErrorDeConfig) as error:
        print(f"\n  ✗ {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  (interrumpido)")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
