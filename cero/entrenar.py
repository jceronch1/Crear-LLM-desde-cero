"""El entrenamiento: cómo unos pesos al azar acaban escribiendo español.

Cada paso del bucle hace cuatro cosas, y no hay nada más:

    1. HACIA DELANTE   pasar un lote por el modelo y medir la *pérdida*: cuánta probabilidad
                       le dio al token que de verdad venía después (entropía cruzada).
    2. HACIA ATRÁS     calcular el *gradiente*: cuánto habría bajado la pérdida al mover un
                       poquito cada peso (`perdida.backward()`).
    3. RECORTAR        si el gradiente es enorme, se reduce para no dar un salto al vacío.
    4. ACTUALIZAR      el optimizador AdamW mueve cada peso un poco en la buena dirección.

El artículo de 2017 entrenó con Adam y una tasa de aprendizaje que sube y luego baja
(§5.3, ecuación 3). Aquí se usa lo que es habitual hoy: AdamW con calentamiento lineal y
descenso en coseno. La idea es la misma: empezar con cuidado, acelerar y frenar al final.

El mismo bucle sirve para las dos etapas: el preentrenamiento (`entrenar_base`) y el ajuste
para conversar (`cero.chat.afinar`). Solo cambian los datos y la tasa de aprendizaje.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from . import dispositivo as disp
from .config import ConfigEntreno, ConfigModelo, guardar_json, miles, ORDEN
from .datos import Lotes
from .generar import completar
from .modelo import GPT
from .proyecto import ErrorDeProyecto, Proyecto, guardar_punto, leer_punto
from .tokenizador import TokenizadorBPE

Avisar = Callable[[str], None]


# ----------------------------------------------------------------------------------------
# Las piezas del bucle
# ----------------------------------------------------------------------------------------
def tasa_de_aprendizaje(paso: int, cfg: ConfigEntreno) -> float:
    """La tasa en cada paso: sube en línea recta durante el calentamiento y baja en coseno.

    Al principio los pesos son ruido y los gradientes, enormes y poco fiables: por eso se
    empieza despacio. Al final conviene afinar con pasos pequeños: por eso se frena.
    """
    if paso < cfg.calentamiento:
        return cfg.lr * (paso + 1) / cfg.calentamiento
    avance = min(1.0, (paso - cfg.calentamiento) / max(1, cfg.pasos - cfg.calentamiento))
    coseno = 0.5 * (1.0 + math.cos(math.pi * avance))        # de 1 a 0, con forma de media onda
    return cfg.lr * (cfg.lr_final + (1.0 - cfg.lr_final) * coseno)


def crear_optimizador(modelo: GPT, cfg: ConfigEntreno, dispositivo: str) -> torch.optim.Optimizer:
    """AdamW. El decaimiento de pesos se aplica solo a las matrices (no a ganancias ni sesgos)."""
    matrices = [p for p in modelo.parameters() if p.requires_grad and p.dim() >= 2]
    resto = [p for p in modelo.parameters() if p.requires_grad and p.dim() < 2]
    grupos = [{"params": matrices, "weight_decay": cfg.decaimiento}, {"params": resto, "weight_decay": 0.0}]
    return torch.optim.AdamW(grupos, lr=cfg.lr, betas=(0.9, 0.95), fused=(dispositivo == "cuda"))


@torch.no_grad()
def medir_perdida(modelo: GPT, lotes: Any, cuantos: int, dispositivo: str) -> float:
    """Pérdida media sobre `cuantos` lotes, sin aprender nada (modo evaluación)."""
    modelo.eval()
    total = 0.0
    for _ in range(cuantos):
        x, y = lotes.siguiente()
        with disp.autocast(dispositivo):
            _, perdida = modelo(x, y)
        total += float(perdida)
    modelo.train()
    return total / cuantos


@torch.enable_grad()
def medir_velocidad(cfg: ConfigModelo, lote: int, dispositivo: str, segundos: float = 3.0) -> dict[str, float]:
    """Cronometra unos pasos de entrenamiento con datos al azar. Devuelve tokens/s y pasos/s.

    Sirve para estimar cuánto tardará una receta en ESTE equipo antes de lanzarla.
    """
    modelo = GPT(cfg).to(dispositivo)
    optimizador = crear_optimizador(modelo, ConfigEntreno(), dispositivo)
    x = torch.randint(0, cfg.vocab, (lote, cfg.contexto), device=dispositivo)

    def paso() -> None:
        optimizador.zero_grad(set_to_none=True)
        with disp.autocast(dispositivo):
            _, perdida = modelo(x, x)
        perdida.backward()
        optimizador.step()

    for _ in range(2):                                       # calentamiento: los primeros pasos son más lentos
        paso()
    disp.sincronizar(dispositivo)
    inicio, hechos = time.perf_counter(), 0
    while time.perf_counter() - inicio < segundos or hechos < 2:
        paso()
        hechos += 1
        disp.sincronizar(dispositivo)
    tiempo = time.perf_counter() - inicio
    return {"pasos_s": hechos / tiempo, "tokens_s": hechos * lote * cfg.contexto / tiempo, "ms_paso": 1000 * tiempo / hechos}


def _duracion(segundos: float) -> str:
    if segundos < 90:
        return f"{segundos:.0f} s"
    if segundos < 5400:
        return f"{segundos / 60:.0f} min"
    return f"{segundos / 3600:.1f} h"


# ----------------------------------------------------------------------------------------
# El bucle
# ----------------------------------------------------------------------------------------
@torch.enable_grad()                 # entrena aunque quien llama haya desactivado los gradientes para mirar
def entrenar(
    modelo: GPT,
    tok: TokenizadorBPE,
    cfg: ConfigEntreno,
    lotes: Any,
    crear_lotes_val: Callable[[], Any],
    carpeta: Path,
    dispositivo: str,
    etapa: str,
    reanudar: bool = False,
    muestra: Callable[[GPT], str] | None = None,
    avisar: Avisar = print,
    progreso: Callable[[int, int, float, float], None] | None = None,
    debe_parar: Callable[[], bool] | None = None,
    ficha: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Entrena `modelo` y deja en `carpeta` el mejor punto, el último y las métricas.

    Se puede interrumpir con Ctrl+C sin perder nada: guarda el estado y se reanuda después
    con `reanudar=True`. Devuelve un resumen (también queda en `estado.json`).
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_ultimo, ruta_mejor = carpeta / "ultimo.pt", carpeta / "mejor.pt"
    ruta_metricas, ruta_estado = carpeta / "metricas.jsonl", carpeta / "estado.json"
    optimizador = crear_optimizador(modelo, cfg, dispositivo)
    paso, mejor_val, segundos_previos, tokens_previos = 0, float("inf"), 0.0, 0

    if reanudar and ruta_ultimo.exists():
        punto = leer_punto(ruta_ultimo, dispositivo)
        if punto["config_modelo"] != modelo.cfg.a_dict():
            raise ErrorDeProyecto(
                "El modelo guardado tiene otra forma que la de config.json, así que no se puede reanudar. "
                "Vuelve a entrenar desde el principio (sin --reanudar) o restaura la configuración."
            )
        modelo.load_state_dict(punto["modelo"])
        optimizador.load_state_dict(punto["optimizador"])
        lotes.restaurar(punto["azar_lotes"])
        torch.set_rng_state(punto["azar_torch"].cpu())
        paso, mejor_val = int(punto["paso"]), float(punto["mejor_val"])
        segundos_previos, tokens_previos = float(punto["segundos"]), int(punto["tokens"])
        avisar(f"  Reanudando desde el paso {miles(paso)}.")
    else:
        ruta_metricas.write_text("", encoding="utf-8")
        ruta_mejor.unlink(missing_ok=True)

    n_parametros = modelo.n_parametros()
    comun = {"config_modelo": modelo.cfg.a_dict(), "config_entreno": cfg.a_dict(), "etapa": etapa,
             "parametros": n_parametros, "version": 1, **(ficha or {})}

    def punto_completo() -> dict[str, Any]:
        return {**comun, "modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(), "paso": paso,
                "mejor_val": mejor_val, "azar_lotes": lotes.estado(), "azar_torch": torch.get_rng_state(),
                "segundos": segundos_previos + time.perf_counter() - inicio, "tokens": tokens_previos + tokens_vistos}

    def resumen(estado: str) -> dict[str, Any]:
        segundos = segundos_previos + time.perf_counter() - inicio
        tokens = tokens_previos + tokens_vistos
        r = {"etapa": etapa, "estado": estado, "paso": paso, "pasos": cfg.pasos,
             "perdida_val": None if math.isinf(mejor_val) else round(mejor_val, 4),
             "perplejidad": None if math.isinf(mejor_val) else round(math.exp(min(mejor_val, 20.0)), 2),
             "segundos": round(segundos, 1), "tokens_vistos": tokens,
             "tokens_s": round(tokens / max(segundos, 1e-9)), "parametros": n_parametros,
             "dispositivo": disp.resumen(dispositivo)}
        if dispositivo == "cuda":
            r["vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        guardar_json(ruta_estado, r)
        return r

    inicio, tokens_vistos, paso_inicial = time.perf_counter(), 0, paso
    if paso >= cfg.pasos:
        avisar(f"  Este entrenamiento ya había terminado ({miles(paso)} pasos). Para seguir, sube «pasos» y usa --reanudar.")
        return resumen("terminado")

    avisar(f"  Modelo de {miles(n_parametros)} parámetros · {disp.resumen(dispositivo)}")
    avisar(f"  {miles(cfg.pasos)} pasos de {cfg.lote * cfg.acumular} secuencias · tasa máxima {cfg.lr:g} · validación cada {cfg.evaluar_cada} pasos")
    avisar("  (Ctrl+C pausa y guarda; después se continúa con --reanudar)")
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

    registrar_cada = max(1, cfg.pasos // 200)
    estado = "terminado"
    with open(ruta_metricas, "a", encoding="utf-8", buffering=1) as registro:
        def anotar(evento: dict[str, Any]) -> None:
            registro.write(json.dumps(evento, ensure_ascii=False) + "\n")

        if paso == 0:                                        # punto de partida: cuánto «sabe» un modelo sin entrenar
            val = medir_perdida(modelo, crear_lotes_val(), cfg.lotes_eval, dispositivo)
            anotar({"tipo": "eval", "paso": 0, "perdida_val": round(val, 4), "segundos": 0.0,
                    "muestra": muestra(modelo) if muestra else None})
            avisar(f"  paso {miles(0):>6}/{miles(cfg.pasos)} · validación {val:.3f}   (azar puro sería ln({modelo.cfg.vocab}) = {math.log(modelo.cfg.vocab):.3f})")

        modelo.train()
        suma_tramo = torch.zeros((), device=dispositivo)     # pérdida acumulada desde la última anotación
        pasos_tramo, tokens_tramo, t_tramo = 0, 0, time.perf_counter()
        suma_eval, pasos_eval = 0.0, 0
        try:
            while paso < cfg.pasos:
                lr = tasa_de_aprendizaje(paso, cfg)
                for grupo in optimizador.param_groups:
                    grupo["lr"] = lr

                optimizador.zero_grad(set_to_none=True)
                for _ in range(cfg.acumular):
                    x, y = lotes.siguiente()
                    with disp.autocast(dispositivo):
                        _, perdida = modelo(x, y)                        # 1. hacia delante
                    (perdida / cfg.acumular).backward()                  # 2. hacia atrás
                    suma_tramo += perdida.detach() / cfg.acumular
                    tokens_tramo += x.numel()
                norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.recorte)   # 3. recortar
                optimizador.step()                                       # 4. actualizar
                paso += 1
                pasos_tramo += 1

                if paso % registrar_cada == 0 or paso % cfg.evaluar_cada == 0 or paso == cfg.pasos:
                    disp.sincronizar(dispositivo)
                    media = float(suma_tramo) / pasos_tramo
                    if not math.isfinite(media):
                        raise ErrorDeProyecto(
                            f"La pérdida dejó de ser un número en el paso {paso} (el entrenamiento «explotó»). "
                            "Casi siempre es una tasa de aprendizaje demasiado alta: baja «lr» en config.json y repite."
                        )
                    ahora = time.perf_counter()
                    velocidad = tokens_tramo / max(ahora - t_tramo, 1e-9)
                    tokens_vistos += tokens_tramo
                    suma_eval += media * pasos_tramo
                    pasos_eval += pasos_tramo
                    anotar({"tipo": "paso", "paso": paso, "perdida": round(media, 4), "lr": round(lr, 7),
                            "norma_grad": round(float(norma), 3), "tokens_s": round(velocidad),
                            "segundos": round(segundos_previos + ahora - inicio, 2)})
                    if progreso is not None:
                        progreso(paso, cfg.pasos, media, velocidad)
                    suma_tramo.zero_()
                    pasos_tramo, tokens_tramo, t_tramo = 0, 0, time.perf_counter()

                if paso % cfg.evaluar_cada == 0 or paso == cfg.pasos:
                    val = medir_perdida(modelo, crear_lotes_val(), cfg.lotes_eval, dispositivo)
                    texto = muestra(modelo) if muestra else None
                    entreno = suma_eval / max(pasos_eval, 1)
                    suma_eval, pasos_eval = 0.0, 0
                    transcurrido = time.perf_counter() - inicio
                    falta = transcurrido / max(paso - paso_inicial, 1) * (cfg.pasos - paso)
                    anotar({"tipo": "eval", "paso": paso, "perdida_val": round(val, 4), "perdida_entreno": round(entreno, 4),
                            "segundos": round(segundos_previos + transcurrido, 2), "muestra": texto})
                    marca = ""
                    if val < mejor_val:
                        mejor_val, marca = val, "  ★ mejor"
                        guardar_punto(ruta_mejor, {**comun, "modelo": modelo.state_dict(), "tokenizador": tok.a_dict(),
                                                    "paso": paso, "perdida_val": val})
                    guardar_punto(ruta_ultimo, punto_completo())
                    avisar(f"  paso {miles(paso):>6}/{miles(cfg.pasos)} · entreno {entreno:.3f} · validación {val:.3f}{marca}"
                           + (f" · faltan ~{_duracion(falta)}" if paso < cfg.pasos else ""))
                    if texto:
                        avisar(f"      «{' '.join(texto.split())[:150]}»")
                    t_tramo = time.perf_counter()            # no contar la evaluación en la velocidad

                if debe_parar is not None and debe_parar():
                    estado = "pausado"
                    break
        except KeyboardInterrupt:
            estado = "pausado"

    if estado == "pausado":
        guardar_punto(ruta_ultimo, punto_completo())
        avisar(f"  Pausado en el paso {miles(paso)}. Todo guardado: continúa con --reanudar.")
    return resumen(estado)


# ----------------------------------------------------------------------------------------
# Etapa «base»: el preentrenamiento
# ----------------------------------------------------------------------------------------
def entrenar_base(
    proyecto: Proyecto,
    dispositivo: str = "auto",
    reanudar: bool = False,
    avisar: Avisar = print,
    progreso: Callable[[int, int, float, float], None] | None = None,
    debe_parar: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Preentrena el modelo del proyecto: aprende a continuar el texto del corpus."""
    dispositivo = disp.elegir(dispositivo)
    disp.preparar(dispositivo)
    config = proyecto.config
    cfg_modelo, cfg = proyecto.config_modelo(), proyecto.config_entreno("base")
    tok = proyecto.tokenizador()
    if cfg_modelo.vocab != tok.vocab:
        raise ErrorDeProyecto(
            f"config.json dice vocab={cfg_modelo.vocab} pero el tokenizador tiene {tok.vocab} tokens. Ejecuta de nuevo:  {ORDEN} datos"
        )
    torch.manual_seed(cfg.semilla)
    modelo = GPT(cfg_modelo).to(dispositivo)
    lotes = Lotes(proyecto.datos / "entreno.bin", cfg_modelo.contexto, cfg.lote, cfg.semilla, dispositivo)

    def lotes_val() -> Lotes:                                # misma semilla = mismos lotes: mediciones comparables
        return Lotes(proyecto.datos / "validacion.bin", cfg_modelo.contexto, cfg.lote, 4321, dispositivo)

    inicio_muestra = config.get("muestra", "Había una vez")

    def muestra(m: GPT) -> str:
        return inicio_muestra + completar(m, tok, inicio_muestra, max_nuevos=50, temperatura=0.8, top_k=40, semilla=7)

    return entrenar(modelo, tok, cfg, lotes, lotes_val, proyecto.etapa("base"), dispositivo, "base",
                    reanudar=reanudar, muestra=muestra, avisar=avisar, progreso=progreso, debe_parar=debe_parar,
                    ficha={"proyecto": proyecto.nombre, "receta": config.get("receta"), "arquitectura": config.get("arquitectura")})


def leer_metricas(carpeta: Path) -> list[dict[str, Any]]:
    """Las mediciones guardadas de una etapa (una por línea en metricas.jsonl)."""
    ruta = carpeta / "metricas.jsonl"
    if not ruta.exists():
        return []
    eventos = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        try:
            eventos.append(json.loads(linea))
        except json.JSONDecodeError:
            continue                                         # una línea a medio escribir: se ignora
    return eventos
