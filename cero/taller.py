"""Apoyo para los cuadernos del curso: el proyecto de clase, las comprobaciones y los dibujos.

Tres cosas, para que los cuadernos se queden solo con lo que hay que aprender:

* `proyecto_del_curso()` y `asegurar()` crean el proyecto «curso» y entrenan lo que falte,
  de modo que cualquier cuaderno se puede abrir sin haber hecho los anteriores.
* `comprobar_...()` corrigen los ejercicios «Tu turno»: comparan lo que has escrito con la
  versión del proyecto y dicen si coincide. Nunca rompen el cuaderno: si algo falla, lo explican.
* `ver_...()` dibujan (mapas de atención, curvas de pérdida...) con matplotlib.

La variable de entorno CERO_RAPIDO=1 encoge todo (pocos pasos, pocos datos): la usan las
pruebas automáticas para ejecutar los trece cuadernos en un par de minutos.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from . import chat, datos, entrenar
from . import modelo as _modelo
from .config import ConfigEntreno, receta
from .proyecto import Proyecto, listar
from .tokenizador import PATRON, TokenizadorBPE, fusionar

RAPIDO = os.environ.get("CERO_RAPIDO") == "1"
NOMBRE_DEL_CURSO = "curso-rapido" if RAPIDO else "curso"


# ========================================================================================
# El proyecto de clase
# ========================================================================================
def proyecto_del_curso(nombre: str | None = None, **cambios: Any) -> Proyecto:
    """Abre el proyecto del curso; si no existe, lo crea con la receta «nano».

    `cambios` permite variar la receta al crear uno nuevo, por ejemplo
    `proyecto_del_curso("fijos", corpus={"fuente": "sintetico", "cuentos": 40000, "nombres": "fijos"})`.
    """
    nombre = nombre or NOMBRE_DEL_CURSO
    if RAPIDO and not nombre.endswith("-rapido"):
        nombre += "-rapido"
    if nombre in listar():
        return Proyecto(nombre)
    config = receta("nano", cambios.pop("arquitectura", "moderna"))
    for clave, valor in cambios.items():
        if isinstance(valor, dict) and isinstance(config.get(clave), dict):
            config[clave].update(valor)
        else:
            config[clave] = valor
    if RAPIDO:                                               # versión de juguete para las pruebas automáticas
        config["corpus"]["cuentos"] = 1500
        config["vocab"] = 420
        config["modelo"].update(vocab=420, contexto=192, d_modelo=48, n_capas=2, n_cabezas=4)
        config["entreno"].update(pasos=24, lote=8, evaluar_cada=12, lotes_eval=2, calentamiento=4)
        config["chat"].update(pasos=12, lote=8, evaluar_cada=6, lotes_eval=2, calentamiento=2)
        config["dialogos"] = 500
    from .proyecto import activar, activo

    anterior = activo()
    proyecto = Proyecto.crear(nombre, config)
    if anterior:                                             # crear este proyecto no cambia cuál es el activo en la consola
        activar(anterior)
    return proyecto


def asegurar(proyecto: Proyecto, hasta: str = "base", dispositivo: str = "auto", avisar: Callable[[str], None] = print) -> Proyecto:
    """Deja el proyecto con las etapas hechas hasta `hasta` («datos», «base» o «chat»).

    Lo que ya esté hecho no se repite. Lo que falte se hace aquí mismo, avisando de que tarda.
    """
    orden = ["datos", "base", "chat"]
    if hasta not in orden:
        raise ValueError(f"«hasta» debe ser una de {orden}.")
    if not (proyecto.datos / "entreno.bin").exists():
        avisar(f"· Preparando los datos del proyecto «{proyecto.nombre}»...")
        datos.preparar(proyecto, lambda _m: None)
    if hasta in ("base", "chat") and not proyecto.mejor("base").exists():
        avisar(f"· Entrenando el modelo base de «{proyecto.nombre}» (unos minutos; se hace una sola vez)...")
        entrenar.entrenar_base(proyecto, dispositivo, avisar=avisar)
    if hasta == "chat" and not proyecto.mejor("chat").exists():
        avisar(f"· Ajustando «{proyecto.nombre}» para conversar (unos minutos; se hace una sola vez)...")
        chat.afinar(proyecto, dispositivo, avisar=avisar)
    return proyecto


# ========================================================================================
# Comprobaciones de los ejercicios
# ========================================================================================
class Correccion:
    """El resultado de un corrector: se puede usar como verdadero/falso (`if comprobar_...(f):`),
    pero en un cuaderno no ensucia la celda con un «Out: True» después del mensaje."""

    def __init__(self, correcto: bool) -> None:
        self.correcto = correcto

    def __bool__(self) -> bool:
        return self.correcto

    def __repr__(self) -> str:
        return "correcto" if self.correcto else "incorrecto"

    def _ipython_display_(self) -> None:          # Jupyter: no mostrar nada (el mensaje ya se imprimió)
        pass


def _corregir(nombre: str, prueba: Callable[[], str | None]) -> Correccion:
    """Ejecuta una prueba que devuelve None si todo va bien o una pista si no."""
    try:
        pista = prueba()
    except NotImplementedError:
        print(f"… {nombre}: todavía sin hacer. Sustituye el `raise NotImplementedError` por tu código.")
        return Correccion(False)
    except Exception as error:  # noqa: BLE001  (el código del estudiante puede fallar de cualquier manera)
        print(f"✗ {nombre}: tu código dio un error ({type(error).__name__}: {error}).")
        return Correccion(False)
    if pista is None:
        print(f"✔ {nombre}: ¡correcto!")
        return Correccion(True)
    print(f"✗ {nombre}: {pista}")
    return Correccion(False)


def bpe_de_libro(texto: str, n_fusiones: int) -> list[tuple[int, int]]:
    """El BPE «de libro», sin ningún truco de velocidad: se recuenta todo en cada vuelta."""
    import re
    from collections import Counter

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
        fusiones.append(mejor)
        trozos = {palabra: fusionar(ids, mejor, 255 + len(fusiones)) for palabra, ids in trozos.items()}
    return fusiones


def comprobar_contar_pares(contar: Callable[[list[int]], dict]) -> Correccion:
    """Clase 1: `contar(ids)` debe devolver cuántas veces aparece cada pareja de vecinos."""
    def prueba() -> str | None:
        ids = [1, 2, 3, 1, 2, 2, 2]
        esperado = {(1, 2): 2, (2, 3): 1, (3, 1): 1, (2, 2): 2}
        obtenido = dict(contar(ids))
        if obtenido != esperado:
            return f"con {ids} debería salir {esperado} y sale {obtenido}."
        return None if dict(contar([7])) == {} else "con una lista de un solo token no hay parejas: debe salir vacío."
    return _corregir("contar parejas", prueba)


def comprobar_fusionar(mi_fusionar: Callable[[list[int], tuple[int, int], int], list[int]]) -> Correccion:
    """Clase 1: `mi_fusionar(ids, par, nuevo)` sustituye cada aparición de `par` por `nuevo`."""
    def prueba() -> str | None:
        casos = [([1, 2, 3, 1, 2], (1, 2), 9, [9, 3, 9]), ([5, 5, 5], (5, 5), 9, [9, 5]), ([1, 2], (2, 1), 9, [1, 2]), ([], (1, 2), 9, [])]
        for ids, par, nuevo, esperado in casos:
            obtenido = list(mi_fusionar(list(ids), par, nuevo))
            if obtenido != esperado:
                return f"fusionar({ids}, {par}, {nuevo}) debería dar {esperado} y da {obtenido}."
        return None
    return _corregir("fusionar una pareja", prueba)


def comprobar_lote(hacer_lote: Callable[[torch.Tensor, list[int], int], tuple[torch.Tensor, torch.Tensor]]) -> Correccion:
    """Clase 2: `hacer_lote(tokens, inicios, contexto)` devuelve (x, y), con y desplazado un token."""
    def prueba() -> str | None:
        tokens = torch.arange(100, 200)
        x, y = hacer_lote(tokens, [0, 10, 50], 8)
        if tuple(x.shape) != (3, 8) or tuple(y.shape) != (3, 8):
            return f"x e y deben tener forma (3, 8); salen {tuple(x.shape)} y {tuple(y.shape)}."
        if x[1].tolist() != list(range(110, 118)):
            return f"la segunda fila de x debería ser los tokens 110..117 y es {x[1].tolist()}."
        if not torch.equal(y[:, :-1], x[:, 1:]) or y[1, -1].item() != 118:
            return "y debe ser x desplazado un lugar: y[i] = el token que viene después de x[i]."
        return None
    return _corregir("construir un lote", prueba)


def comprobar_sinusoidal(mi_tabla: Callable[[int, int], torch.Tensor]) -> Correccion:
    """Clase 3: la codificación posicional del artículo, PE(pos, 2i) = sen(...), PE(pos, 2i+1) = cos(...)."""
    def prueba() -> str | None:
        tabla = torch.as_tensor(mi_tabla(40, 16), dtype=torch.float32)
        if tuple(tabla.shape) != (40, 16):
            return f"la tabla debe tener forma (40, 16) y tiene {tuple(tabla.shape)}."
        diferencia = float((tabla - _modelo.tabla_sinusoidal(40, 16)).abs().max())
        if diferencia > 1e-4:
            if float((tabla[:, 0::2] - _modelo.tabla_sinusoidal(40, 16)[:, 1::2]).abs().max()) < 1e-4:
                return "has cambiado el sitio del seno y el coseno: el seno va en las columnas pares."
            return f"no coincide con la fórmula del artículo (diferencia máxima {diferencia:.3f}). Revisa el exponente 2i/d."
        return None
    return _corregir("codificación sinusoidal", prueba)


def comprobar_mascara(mi_mascara: Callable[[int], torch.Tensor]) -> Correccion:
    """Clase 4: `mi_mascara(T)` es una matriz T x T booleana: True donde se permite mirar."""
    def prueba() -> str | None:
        m = torch.as_tensor(mi_mascara(5)).bool()
        if tuple(m.shape) != (5, 5):
            return f"debe tener forma (5, 5) y tiene {tuple(m.shape)}."
        if not torch.equal(m, _modelo.mascara_causal(5, 5)):
            if torch.equal(m, ~_modelo.mascara_causal(5, 5)):
                return "está al revés: True debe marcar lo que SÍ se puede mirar (el propio token y los anteriores)."
            if torch.equal(m, _modelo.mascara_causal(5, 5).T):
                return "está traspuesta: la fila es quien pregunta y la columna a quién mira; se mira hacia atrás."
            return "no es el triángulo inferior: cada token mira a sí mismo y a todos los anteriores, y a ninguno posterior."
        return None
    return _corregir("máscara causal", prueba)


def comprobar_atencion(mi_atencion: Callable[..., Any]) -> Correccion:
    """Clase 4: `mi_atencion(q, k, v, mascara)` debe dar lo mismo que la de PyTorch."""
    def prueba() -> str | None:
        torch.manual_seed(0)
        q, k, v = torch.randn(3, 2, 6, 8).unbind(0)
        mascara = _modelo.mascara_causal(6, 6)
        salida = mi_atencion(q, k, v, mascara)
        salida = salida[0] if isinstance(salida, tuple) else salida
        referencia = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if tuple(salida.shape) != tuple(referencia.shape):
            return f"la salida debe tener forma {tuple(referencia.shape)} y tiene {tuple(salida.shape)}."
        diferencia = float((salida - referencia).abs().max())
        if diferencia < 1e-4:
            return None
        sin_escala, _ = _atencion_variante(q, k, v, mascara, escala=False)
        if float((salida - sin_escala).abs().max()) < 1e-4:
            return "te falta dividir entre la raíz de d_k (la dimensión de las consultas) antes del softmax."
        sin_mascara = F.scaled_dot_product_attention(q, k, v)
        if float((salida - sin_mascara).abs().max()) < 1e-4:
            return "no estás usando la máscara: los tokens están mirando al futuro."
        return f"no coincide con la referencia (diferencia máxima {diferencia:.3f}). Repasa: puntos = Q·Kᵀ/√d, máscara, softmax, pesos·V."
    return _corregir("atención de producto escalar", prueba)


def _atencion_variante(q, k, v, mascara, escala: bool = True):
    puntos = q @ k.transpose(-2, -1)
    if escala:
        puntos = puntos / math.sqrt(q.size(-1))
    pesos = puntos.masked_fill(~mascara, float("-inf")).softmax(-1)
    return pesos @ v, pesos


def comprobar_norma_rms(mi_norma: Callable[[torch.Tensor], torch.Tensor]) -> Correccion:
    """Clase 5: `mi_norma(x)` divide cada vector entre la raíz de la media de sus cuadrados."""
    def prueba() -> str | None:
        torch.manual_seed(1)
        x = torch.randn(4, 7, 32) * 5 + 2
        y = mi_norma(x)
        if tuple(y.shape) != tuple(x.shape):
            return "la salida debe tener la misma forma que la entrada."
        esperado = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if float((y - esperado).abs().max()) < 1e-3:
            return None
        if float((y - F.layer_norm(x, (32,))).abs().max()) < 1e-3:
            return "eso es LayerNorm (has restado la media). RMSNorm no la resta: solo divide entre la raíz de la media de x²."
        return "la media de los cuadrados de cada vector de salida debería ser 1. Usa mean(dim=-1, keepdim=True)."
    return _corregir("RMSNorm", prueba)


def comprobar_tasa(mi_tasa: Callable[[int], float], cfg: ConfigEntreno | None = None) -> Correccion:
    """Clase 7: `mi_tasa(paso)` = calentamiento lineal y después descenso en coseno."""
    cfg = cfg or ConfigEntreno(pasos=1000, lr=1e-3, calentamiento=100, lr_final=0.1)

    def prueba() -> str | None:
        for paso in (0, 50, 99, 100, 550, 999):
            esperado, obtenido = entrenar.tasa_de_aprendizaje(paso, cfg), float(mi_tasa(paso))
            if abs(esperado - obtenido) > 1e-7:
                tramo = "el calentamiento (debe subir en línea recta hasta lr)" if paso < cfg.calentamiento else "el coseno"
                return f"en el paso {paso} debería valer {esperado:.6f} y vale {obtenido:.6f}. Revisa {tramo}."
        return None
    return _corregir("tasa de aprendizaje", prueba)


def comprobar_temperatura(aplicar: Callable[[torch.Tensor, float], torch.Tensor]) -> Correccion:
    """Clase 9: `aplicar(logits, temperatura)` devuelve probabilidades = softmax(logits / temperatura)."""
    def prueba() -> str | None:
        logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
        for t in (0.5, 1.0, 2.0):
            probs = torch.as_tensor(aplicar(logits, t), dtype=torch.float32)
            if abs(float(probs.sum()) - 1.0) > 1e-4:
                return "la salida deben ser probabilidades: tienen que sumar 1 (¿falta el softmax?)."
            if float((probs - (logits / t).softmax(-1)).abs().max()) > 1e-4:
                return f"con temperatura {t} no coincide. Hay que DIVIDIR los logits entre la temperatura antes del softmax."
        return None
    return _corregir("temperatura", prueba)


def comprobar_top_k(mi_top_k: Callable[[torch.Tensor, int], torch.Tensor]) -> Correccion:
    """Clase 9: `mi_top_k(logits, k)` deja a -infinito todo menos los k logits más altos."""
    def prueba() -> str | None:
        logits = torch.tensor([0.1, 3.0, -1.0, 2.0, 0.5, 1.5])
        salida = torch.as_tensor(mi_top_k(logits.clone(), 3), dtype=torch.float32)
        vivos = torch.isfinite(salida)
        if int(vivos.sum()) != 3:
            return f"deben sobrevivir exactamente 3 tokens y sobreviven {int(vivos.sum())}."
        if vivos.tolist() != [False, True, False, True, False, True]:
            return "han sobrevivido tokens que no eran los 3 más probables."
        if float((salida[vivos] - logits[vivos]).abs().max()) > 1e-6:
            return "los logits que sobreviven no deben cambiar de valor."
        return None
    return _corregir("top-k", prueba)


def comprobar_plantilla(mi_plantilla: Callable[[TokenizadorBPE, list[dict[str, str]]], tuple[list[int], list[int]]]) -> Correccion:
    """Clase 10: `mi_plantilla(tok, mensajes)` devuelve (tokens, máscara) como `cero.chat.codificar_dialogo`."""
    def prueba() -> str | None:
        from .cuentos import generar_cuentos

        tok = TokenizadorBPE.entrenar(list(generar_cuentos(200, 1)), 400)
        mensajes = [chat.u("Hola"), chat.a("¡Hola! Soy Cero."), chat.u("¿Cuánto es 1 + 1?"), chat.a("1 + 1 = 2.")]
        ids, mascara = mi_plantilla(tok, mensajes)
        bien_ids, bien_mascara = chat.codificar_dialogo(tok, mensajes)
        if list(ids) != bien_ids:
            return ("los tokens no coinciden. Debe ser: <|inicio|>, y por cada mensaje <|usuario|> o <|asistente|>, su texto y <|fin|>.\n"
                    f"   tuyo:     {tok.decodificar(ids)}\n   esperado: {tok.decodificar(bien_ids)}")
        if list(mascara) != bien_mascara:
            aprendido = tok.decodificar([i for i, m in zip(ids, mascara) if m])
            return f"la máscara no coincide. Con la tuya se aprendería «{aprendido}»; solo debe aprenderse lo que dice el asistente y su <|fin|>."
        return None
    return _corregir("plantilla de chat", prueba)


# ========================================================================================
# Dibujos (matplotlib se importa aquí dentro: el resto del proyecto no lo necesita)
# ========================================================================================
def _plt():
    import matplotlib.pyplot as plt

    return plt


def ver_atencion(pesos: Any, tokens: list[str], titulo: str = "", ax: Any = None) -> None:
    """Mapa de calor de una matriz de atención (fila = quien pregunta, columna = a quién mira)."""
    plt = _plt()
    propio = ax is None
    if propio:
        lado = max(4.0, 0.42 * len(tokens))
        _, ax = plt.subplots(figsize=(lado, lado))
    matriz = torch.as_tensor(pesos, dtype=torch.float32).detach().cpu()
    ax.imshow(matriz, cmap="Greens", vmin=0, vmax=max(float(matriz.max()), 1e-6))
    rotulos = [t.replace("\n", "↵") for t in tokens]
    ax.set_xticks(range(len(tokens)), rotulos, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)), rotulos, fontsize=8)
    ax.set_xlabel("a quién mira")
    ax.set_ylabel("quién pregunta")
    ax.set_title(titulo, fontsize=10)
    if propio:
        plt.tight_layout()
        plt.show()


def ver_curvas(eventos: list[dict[str, Any]], vocab: int | None = None, titulo: str = "Pérdida") -> None:
    """Curvas de entrenamiento y validación a partir de las mediciones guardadas."""
    plt = _plt()
    pasos = [e for e in eventos if e["tipo"] == "paso"]
    evals = [e for e in eventos if e["tipo"] == "eval"]
    plt.figure(figsize=(8, 3.6))
    if pasos:
        plt.plot([e["paso"] for e in pasos], [e["perdida"] for e in pasos], color="0.6", lw=1, label="entrenamiento")
    if evals:
        plt.plot([e["paso"] for e in evals], [e["perdida_val"] for e in evals], "o-", color="#0f6e66", lw=2, label="validación")
    if vocab:
        plt.axhline(math.log(vocab), color="#a8530b", ls="--", lw=1, label=f"azar: ln({vocab}) = {math.log(vocab):.2f}")
    plt.xlabel("paso")
    plt.ylabel("pérdida")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def ver_tabla(tabla: Any, titulo: str = "", etiqueta_x: str = "dimensión", etiqueta_y: str = "posición") -> None:
    """Una matriz cualquiera como mapa de calor (por ejemplo, la tabla de posiciones)."""
    plt = _plt()
    plt.figure(figsize=(8, 3.4))
    plt.imshow(torch.as_tensor(tabla, dtype=torch.float32).detach().cpu(), aspect="auto", cmap="RdBu")
    plt.colorbar()
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def ver_candidatos(lista: list[dict[str, Any]], titulo: str = "¿Qué token viene ahora?") -> None:
    """Barras con la probabilidad original y la ajustada de cada candidato (ver cero.generar.candidatos)."""
    plt = _plt()
    nombres = [repr(c["texto"])[1:-1] for c in lista][::-1]
    plt.figure(figsize=(7, 0.32 * len(lista) + 1))
    plt.barh(nombres, [c["prob"] for c in lista][::-1], color="#d9eeeb", edgecolor="#0f6e66", label="lo que dice el modelo")
    plt.barh(nombres, [c["prob_ajustada"] for c in lista][::-1], height=0.35, color="#0f6e66", label="tras los mandos")
    plt.xlabel("probabilidad")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def semilla(numero: int = 0) -> None:
    """Fija el azar de Python y de PyTorch para que una celda dé siempre lo mismo."""
    random.seed(numero)
    torch.manual_seed(numero)


# ========================================================================================
# Entrenamientos de bolsillo para los experimentos de clase
# ========================================================================================
def tokens_de(textos: list[str], tok: TokenizadorBPE) -> torch.Tensor:
    """Todos los textos seguidos, como un tensor de tokens (cada uno empieza con <|inicio|>)."""
    inicio = tok.id("<|inicio|>")
    return torch.tensor([i for texto in textos for i in [inicio] + tok.codificar_texto(texto)], dtype=torch.long)


def lote_de(tokens: torch.Tensor, contexto: int, lote: int, generador: torch.Generator, dispositivo: str = "cpu"):
    """Un lote (x, y) de ventanas al azar sacadas de un tensor de tokens."""
    inicios = torch.randint(0, len(tokens) - contexto - 1, (lote,), generator=generador)
    x = torch.stack([tokens[i : i + contexto] for i in inicios])
    y = torch.stack([tokens[i + 1 : i + contexto + 1] for i in inicios])
    return x.to(dispositivo), y.to(dispositivo)


@torch.enable_grad()
def entrenar_mini(modelo: _modelo.GPT, entreno: torch.Tensor, validacion: torch.Tensor | None = None, pasos: int = 200,
                  lote: int = 16, lr: float = 3e-3, medir_cada: int = 20, dispositivo: str = "cpu", semilla_: int = 0) -> list[dict[str, float]]:
    """Un entrenamiento corto y a la vista, para experimentos de clase. Devuelve las mediciones.

    Es el mismo bucle que `cero.entrenar`, sin guardar nada en disco: sirve para comparar
    en un minuto dos tasas de aprendizaje, dos arquitecturas o dos cantidades de datos.
    """
    cfg = ConfigEntreno(pasos=pasos, lote=lote, lr=lr, calentamiento=max(1, pasos // 20))
    modelo.to(dispositivo).train()
    optimizador = entrenar.crear_optimizador(modelo, cfg, dispositivo)
    azar = torch.Generator().manual_seed(semilla_)
    contexto = modelo.cfg.contexto
    historia: list[dict[str, float]] = []
    for paso in range(pasos + 1):
        if paso % medir_cada == 0 or paso == pasos:
            medida = {"paso": paso}
            modelo.eval()
            with torch.no_grad():
                fijo = torch.Generator().manual_seed(123)
                medida["entreno"] = sum(float(modelo(*lote_de(entreno, contexto, lote, fijo, dispositivo))[1]) for _ in range(4)) / 4
                if validacion is not None:
                    medida["validacion"] = sum(float(modelo(*lote_de(validacion, contexto, lote, fijo, dispositivo))[1]) for _ in range(4)) / 4
            modelo.train()
            historia.append(medida)
            if not math.isfinite(medida["entreno"]):
                break
        if paso == pasos:
            break
        for grupo in optimizador.param_groups:
            grupo["lr"] = entrenar.tasa_de_aprendizaje(paso, cfg)
        x, y = lote_de(entreno, contexto, lote, azar, dispositivo)
        _, perdida = modelo(x, y)
        optimizador.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.recorte)
        optimizador.step()
    modelo.eval()
    return historia


def ver_historias(historias: dict[str, list[dict[str, float]]], clave: str = "validacion", titulo: str = "") -> None:
    """Dibuja juntas varias curvas devueltas por `entrenar_mini`, una por nombre."""
    plt = _plt()
    plt.figure(figsize=(8, 3.6))
    for nombre, historia in historias.items():
        if historia and clave in historia[0]:
            plt.plot([m["paso"] for m in historia], [m[clave] for m in historia], "o-", ms=3, label=nombre)
    plt.xlabel("paso")
    plt.ylabel(f"pérdida de {clave}")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
