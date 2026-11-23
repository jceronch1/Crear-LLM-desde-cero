"""De LLM a chatbot: la plantilla de conversación y el ajuste fino supervisado (SFT).

El modelo «base» solo sabe continuar texto: si le escribes una pregunta, sigue escribiendo
como si fuera un cuento. Un chatbot es EL MISMO modelo, entrenado un poco más con textos que
tienen forma de conversación. No cambia la arquitectura ni una línea del bucle; cambian dos cosas:

1. LA PLANTILLA. Cada conversación se escribe como una sola secuencia de tokens, con los
   tokens especiales marcando quién habla:

       <|inicio|><|usuario|>Hola<|fin|><|asistente|>¡Hola! Soy Cero.<|fin|>

2. LA MÁSCARA. Solo se calcula la pérdida en lo que dice el asistente (y en su <|fin|>).
   No queremos que el modelo aprenda a escribir preguntas, sino a responderlas y a callar
   al terminar:

       tokens:    <|inicio|> <|usuario|> Hola <|fin|> <|asistente|> ¡Hola! Soy Cero. <|fin|>
       ¿se aprende?   no         no       no    no         no         sí   sí   sí     sí

Para conversar se escribe la conversación hasta `<|asistente|>` y se deja que el modelo
continúe hasta que él mismo escriba `<|fin|>`.

Este archivo también fabrica las conversaciones de entrenamiento. Nadie las ha escrito a
mano una a una: salen de reglas sobre el propio corpus (y eso es una limitación que se
discute en la clase 11).
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import dispositivo as disp
from .config import guardar_json, miles, ORDEN
from .cuentos import TIPOS, Cuento, cuento
from .datos import documentos, huella
from .entrenar import entrenar
from .generar import generar
from .modelo import GPT
from .proyecto import ErrorDeProyecto, Proyecto, cargar_modelo
from .tokenizador import ASISTENTE, FIN, INICIO, USUARIO, TokenizadorBPE

Avisar = Callable[[str], None]
Mensaje = dict[str, str]          # {"rol": "usuario" | "asistente", "texto": "..."}
NOMBRE = "Cero"                   # cómo se llama nuestro asistente


# ========================================================================================
# 1. La plantilla
# ========================================================================================
def codificar_dialogo(tok: TokenizadorBPE, mensajes: list[Mensaje], abrir_respuesta: bool = False) -> tuple[list[int], list[int]]:
    """Convierte una conversación en (tokens, máscara). La máscara vale 1 donde se aprende.

    Con `abrir_respuesta=True` la secuencia termina en `<|asistente|>`: es la forma de pedirle
    al modelo que conteste.

    El texto de los mensajes se codifica con `codificar_texto`, que NO reconoce tokens
    especiales: si alguien escribe «<|fin|>» en su mensaje, son caracteres normales y no
    puede hacerse pasar por el asistente.
    """
    ids, mascara = [tok.id(INICIO)], [0]
    for mensaje in mensajes:
        contenido = tok.codificar_texto(mensaje["texto"])
        if mensaje["rol"] == "usuario":
            ids += [tok.id(USUARIO)] + contenido + [tok.id(FIN)]
            mascara += [0] * (len(contenido) + 2)
        elif mensaje["rol"] == "asistente":
            ids += [tok.id(ASISTENTE)] + contenido + [tok.id(FIN)]
            mascara += [0] + [1] * (len(contenido) + 1)      # se aprende la respuesta y su <|fin|>
        else:
            raise ValueError(f"Rol desconocido: {mensaje['rol']!r} (usa «usuario» o «asistente»).")
    if abrir_respuesta:
        ids.append(tok.id(ASISTENTE))
        mascara.append(0)
    return ids, mascara


def u(texto: str) -> Mensaje:
    return {"rol": "usuario", "texto": texto}


def a(texto: str) -> Mensaje:
    return {"rol": "asistente", "texto": texto}


# ========================================================================================
# 2. Las conversaciones de entrenamiento
# ========================================================================================
CHARLA: dict[str, tuple[list[str], list[str]]] = {
    "saludo": (
        ["Hola", "¡Hola!", "Hola, ¿qué tal?", "Buenos días", "Buenas tardes", "Buenas noches", "Buenas", f"Hola, {NOMBRE}"],
        [f"¡Hola! Soy {NOMBRE}. ¿Quieres que te cuente un cuento?", "¡Hola! ¿Te cuento un cuento?",
         "¡Hola! ¿Sobre qué animal quieres un cuento?", "¡Hola! Me alegra verte. Puedo contarte un cuento."],
    ),
    "identidad": (
        ["¿Quién eres?", "¿Cómo te llamas?", "¿Cuál es tu nombre?", "¿Qué eres?", "¿Eres una persona?", "¿Eres un robot?"],
        [f"Me llamo {NOMBRE}. Soy un modelo de lenguaje muy pequeño que cuenta cuentos.",
         f"Soy {NOMBRE}, un modelo de lenguaje pequeño. Me entrenaron desde cero para contar cuentos.",
         f"Soy {NOMBRE}. No soy una persona: soy un programa que aprendió a escribir cuentos."],
    ),
    "capacidades": (
        ["¿Qué sabes hacer?", "¿Qué puedes hacer?", "¿En qué me puedes ayudar?", "¿Para qué sirves?", "¿Qué haces?"],
        ["Sé contar cuentos cortos, responder preguntas sobre un cuento y hacer sumas pequeñas.",
         "Puedo contarte un cuento sobre el animal que elijas. También sé sumar números pequeños.",
         "Cuento cuentos y sumo números pequeños. Pídeme un cuento sobre un animal."],
    ),
    "estado": (
        ["¿Cómo estás?", "¿Qué tal estás?", "¿Estás bien?", "¿Cómo te va?"],
        ["Estoy muy bien, gracias. ¿Quieres un cuento?", "¡Muy bien! Tengo ganas de contar un cuento."],
    ),
    "gracias": (
        ["Gracias", "¡Muchas gracias!", f"Gracias, {NOMBRE}", "Qué bonito, gracias", "Me gustó mucho"],
        ["¡De nada! ¿Quieres otro cuento?", "¡Con mucho gusto!", "Me alegra que te guste. ¿Te cuento otro?"],
    ),
    "despedida": (
        ["Adiós", "Hasta luego", "Chao", "Nos vemos", "Hasta mañana", "Me voy"],
        ["¡Adiós! Vuelve cuando quieras otro cuento.", "¡Hasta luego! Que tengas un buen día.", "¡Nos vemos pronto!"],
    ),
}

# Preguntas que el modelo NO puede saber. Se le enseña a decirlo en vez de inventar.
FUERA_DE_ALCANCE = [
    "¿Cuál es la capital de Francia?", "¿Quién descubrió América?", "¿Qué hora es?", "¿Qué tiempo hace hoy?",
    "¿Cuántos años tienes?", "Explícame la física cuántica.", "¿Quién ganó el partido?", "Escribe un programa en Python.",
    "¿Cuánto cuesta un coche?", "¿Dónde está Japón?", "¿Qué es un átomo?", "Traduce esto al inglés.",
    "¿Quién es el presidente?", "¿Cómo se hace una paella?", "Háblame de la historia de Roma.",
    "¿Cuál es el río más largo del mundo?", "¿Qué día es hoy?", "Recomiéndame una película.", "¿Cuánto mide la Luna?",
    "Dame una receta de pan.", "¿Qué es la democracia?", "¿Cuántos planetas hay?", "¿Quién escribió el Quijote?",
    "¿Qué es internet?", "¿Cómo funciona un motor?", "¿Cuál es la montaña más alta?", "Resuelve esta ecuación.",
    "¿Qué noticias hay hoy?", "¿Cuál es tu color favorito?", "¿Dónde vives?",
]
# Estas no se usan para entrenar: sirven para evaluar si generaliza el «no lo sé».
FUERA_DE_ALCANCE_NUEVAS = [
    "¿Cuál es la capital de Italia?", "¿Quién inventó el teléfono?", "¿Cuánto pesa un elefante?",
    "Explícame qué es un volcán.", "¿Qué idioma se habla en Brasil?", "¿Cuándo empezó la guerra?",
    "Dame un consejo para estudiar.", "¿Qué es un agujero negro?", "¿Quién pintó ese cuadro?", "¿Cómo se llega a Madrid?",
]
NO_SE = [
    "Eso no lo sé. Soy un modelo muy pequeño: solo sé contar cuentos y sumar un poco.",
    "No lo sé. Solo aprendí cuentos cortos. ¿Quieres que te cuente uno?",
    "Lo siento, eso no lo sé. Puedo contarte un cuento.",
]

PEDIR_CUENTO = [
    "Cuéntame un cuento sobre {quien}.", "Cuéntame un cuento de {quien}.", "Escribe un cuento sobre {quien}.",
    "¿Me cuentas un cuento sobre {quien}?", "Quiero un cuento de {quien}.", "Inventa un cuento con {quien}.",
]
PEDIR_CUALQUIERA = ["Cuéntame un cuento.", "¿Me cuentas un cuento?", "Quiero un cuento.", "Escribe un cuento.", "Cuéntame otro cuento."]
SUMA_MAXIMA = 10                  # se suman números de 0 a 10: 121 sumas posibles


def suma_reservada(x: int, y: int) -> bool:
    """Una de cada siete sumas no se enseña nunca: sirve para ver si el modelo «sabe sumar»
    o solo recuerda las que vio."""
    return (x + 2 * y) % 7 == 3


def _charla(rng: random.Random, tema: str | None = None) -> list[Mensaje]:
    preguntas, respuestas = CHARLA[tema or rng.choice(list(CHARLA))]
    return [u(rng.choice(preguntas)), a(rng.choice(respuestas))]


def _no_se(rng: random.Random) -> list[Mensaje]:
    return [u(rng.choice(FUERA_DE_ALCANCE)), a(rng.choice(NO_SE))]


def _suma(rng: random.Random) -> list[Mensaje]:
    while True:
        x, y = rng.randint(0, SUMA_MAXIMA), rng.randint(0, SUMA_MAXIMA)
        if not suma_reservada(x, y):
            break
    pregunta = rng.choice([f"¿Cuánto es {x} + {y}?", f"¿Cuánto es {x} más {y}?", f"Suma {x} y {y}.", f"{x} + {y}"])
    return [u(pregunta), a(f"{x} + {y} = {x + y}.")]


def peticion_de_cuento(rng: random.Random, c: Cuento) -> str:
    """Una forma de pedir el cuento `c`: a veces solo el personaje, a veces también el nombre."""
    if rng.random() < 0.10:
        return rng.choice(PEDIR_CUALQUIERA)
    p = c.protagonista
    quien = p.un_tipo
    if rng.random() < 0.6:
        quien += rng.choice([f" llamad{p.o} {p.nombre}", f" que se llama {p.nombre}"])
    return rng.choice(PEDIR_CUENTO).format(quien=quien)


def dialogos_sinteticos(cuantos: int, semilla: int, variados: bool = True,
                        cabe: Callable[[list[Mensaje]], bool] | None = None) -> Iterator[tuple[str, list[Mensaje]]]:
    """Conversaciones sobre el mundo de cuentos sintéticos. Entrega (categoría, mensajes).

    Cada conversación tiene de uno a tres intercambios, como una charla de verdad: a veces
    empieza con un saludo, a veces tras el cuento llega una pregunta sobre él, a veces
    termina con un «gracias». `cabe` dice si la conversación entra en la ventana del modelo;
    si no entra, se le quitan intercambios por el final.
    """
    rng = random.Random(semilla)
    for _ in range(cuantos):
        turnos: list[list[Mensaje]] = []
        if rng.random() < 0.15:
            turnos.append(_charla(rng, "saludo"))
        azar = rng.random()
        if azar < 0.38:                                      # pedir un cuento
            c = cuento(rng, variados=variados)
            turnos.append([u(peticion_de_cuento(rng, c)), a(c.texto)])
            categoria = "cuento"
            if rng.random() < 0.45:                          # ...y preguntar después por lo que pasó en él
                for pregunta, contestacion in rng.sample(c.hechos, rng.choice([1, 2])):
                    turnos.append([u(pregunta), a(contestacion)])
                categoria = "cuento_y_pregunta"
        elif azar < 0.62:                                    # leer un cuento que trae el usuario y responder
            c = cuento(rng, variados=variados)
            # Varias preguntas sobre el mismo cuento: el texto se «paga» una vez y se aprende de todas.
            preguntas = rng.sample(c.hechos, rng.choice([1, 2, 2, 3]))
            turnos.append([u(f"{c.texto}\n\n{preguntas[0][0]}"), a(preguntas[0][1])])
            for pregunta, contestacion in preguntas[1:]:
                turnos.append([u(pregunta), a(contestacion)])
            categoria = "lectura"
        elif azar < 0.82:
            turnos.append(_suma(rng))
            if rng.random() < 0.5:
                turnos.append(_suma(rng))
            categoria = "suma"
        elif azar < 0.91:
            turnos.append(_no_se(rng))
            categoria = "no_se"
        else:
            turnos.append(_charla(rng))
            categoria = "charla"
        if rng.random() < 0.12:
            turnos.append(_charla(rng, rng.choice(["gracias", "despedida"])))
        while cabe is not None and len(turnos) > 1 and not cabe([m for t in turnos for m in t]):
            turnos.pop()
        yield categoria, [m for t in turnos for m in t]


# ---------------------------------------------------------------------------- corpus reales
_FRASE = re.compile(r"[^.!?…]+[.!?…]+[\"»”')]*\s*")
_NOMBRE_PROPIO = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,12}"
_LLAMADO = re.compile(rf"\b(?:llamad[oa]|se llamaba|de nombre) ({_NOMBRE_PROPIO})\b")
_TIPOS_REALES = dict(TIPOS) | {
    "pez": "m", "cerdo": "m", "cerdito": "m", "gatito": "m", "perrito": "m", "osito": "m", "tigre": "m", "pollito": "m",
    "muñeca": "f", "hada": "f", "bruja": "f", "ratona": "f", "perrita": "f", "gatita": "f", "sirena": "f",
}
_TIPO = re.compile(r"\b(un|una) (?:pequeñ[oa] |gran |joven |viej[oa] |pobre |hermos[oa] )?("
                   + "|".join(sorted(_TIPOS_REALES, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def frases(texto: str) -> list[str]:
    return [f.strip() for f in _FRASE.findall(texto) if f.strip()]


def recortar(tok: TokenizadorBPE, texto: str, max_tokens: int) -> str:
    """Las primeras frases completas de `texto` que caben en `max_tokens`."""
    salida, usados = [], 0
    for frase in frases(texto.replace("\n", " ")):
        n = len(tok.codificar_texto(" " + frase))
        if usados + n > max_tokens:
            break
        salida.append(frase)
        usados += n
    return " ".join(salida)


def dialogos_reales(historias: list[str], cuantos: int, semilla: int, tok: TokenizadorBPE,
                    contexto: int) -> Iterator[tuple[str, list[Mensaje]]]:
    """Conversaciones a partir de cuentos reales. Las reglas miran solo la PRIMERA frase,
    que es donde estos cuentos presentan a su protagonista."""
    rng = random.Random(semilla)
    presupuesto = contexto - 48                              # sitio para la petición y los tokens especiales
    for _ in range(cuantos):
        azar = rng.random()
        if azar < 0.10:
            yield "charla", _charla(rng)
        elif azar < 0.17:
            yield "no_se", _no_se(rng)
        elif azar < 0.27:
            yield "suma", _suma(rng) + (_suma(rng) if rng.random() < 0.5 else [])
        else:
            historia = rng.choice(historias)
            partes = frases(historia.replace("\n", " "))
            if len(partes) < 4:
                continue
            saludo = _charla(rng, "saludo") if rng.random() < 0.12 else []
            if azar < 0.45:                                  # «Continúa este cuento: ...»
                comienzo = " ".join(partes[:2])
                resto = recortar(tok, " ".join(partes[2:]), presupuesto - len(tok.codificar_texto(comienzo)) - 24 * bool(saludo))
                if len(resto) > 80:
                    yield "continuar", saludo + [u(f"Continúa este cuento: {comienzo}"), a(resto)]
                continue
            cuerpo = recortar(tok, historia, presupuesto - 24 * bool(saludo))
            if len(cuerpo) < 80:
                continue
            tipo = _TIPO.search(partes[0])
            if tipo is None:                                 # sin protagonista claro: petición genérica
                if rng.random() < 0.15:
                    yield "cuento", saludo + [u(rng.choice(PEDIR_CUALQUIERA)), a(cuerpo)]
                continue
            quien = f"{tipo.group(1).lower()} {tipo.group(2).lower()}"
            nombre = _LLAMADO.search(" ".join(partes[:2]))
            if nombre and rng.random() < 0.6:
                genero = _TIPOS_REALES[tipo.group(2).lower()]
                quien += f" llamad{'o' if genero == 'm' else 'a'} {nombre.group(1)}"
            yield "cuento", saludo + [u(rng.choice(PEDIR_CUENTO).format(quien=quien)), a(cuerpo)]


def textos_para_tokenizador(corpus: dict[str, Any], semilla: int, cuantos: int = 6000) -> Iterator[str]:
    """Frases de conversación para que el tokenizador las conozca (los cuentos ya los ve aparte)."""
    rng = random.Random(semilla + 77)
    variados = corpus.get("nombres", "mixtos") != "fijos"
    for _ in range(cuantos):
        azar = rng.random()
        if azar < 0.35:
            c = cuento(rng, variados=variados)
            pregunta, contestacion = rng.choice(c.hechos)
            mensajes = [u(peticion_de_cuento(rng, c)), a(f"{pregunta} {contestacion}")]
        elif azar < 0.60:
            mensajes = _suma(rng)
        elif azar < 0.75:
            mensajes = _no_se(rng)
        else:
            mensajes = _charla(rng)
        yield " ".join(m["texto"] for m in mensajes)


def preparar_dialogos(proyecto: Proyecto, avisar: Avisar = print) -> dict[str, Any]:
    """Fabrica las conversaciones, las convierte en tokens y las guarda en `datos/chat_*.npz`."""
    config = proyecto.config
    tok = proyecto.tokenizador()
    contexto = int(config["modelo"]["contexto"])
    cuantos = int(config.get("dialogos", 30_000))
    semilla = int(config["chat"].get("semilla", 1337)) + 1
    fuente = config["corpus"]["fuente"]
    inicio = time.time()
    avisar(f"  Fabricando {miles(cuantos)} conversaciones a partir del corpus «{fuente}»...")

    def cabe(mensajes: list[Mensaje]) -> bool:
        return len(codificar_dialogo(tok, mensajes)[0]) <= contexto

    if fuente == "sintetico":
        variados = config["corpus"].get("nombres", "mixtos") != "fijos"
        origen = dialogos_sinteticos(cuantos * 2, semilla, variados, cabe)
    else:
        # Solo cuentos de la parte de entrenamiento: la validación del corpus sigue intacta.
        limite, historias = 60_000, []
        fraccion = float(config["corpus"].get("validacion", 0.1 if fuente == "texto" else 0.02))
        for texto in documentos(config["corpus"], semilla, avisar):
            if huella(texto) % 10_000 >= int(fraccion * 10_000):
                historias.append(texto)
            if len(historias) >= limite:
                break
        origen = dialogos_reales(historias, cuantos * 3, semilla, tok, contexto)

    partes: dict[str, dict[str, list]] = {p: {"tokens": [], "mascara": [], "limites": [0]} for p in ("entreno", "validacion")}
    categorias: dict[str, int] = {}
    ejemplos: list[dict[str, Any]] = []
    largos, descartados, hechos = 0, 0, 0
    for categoria, mensajes in origen:
        ids, mascara = codificar_dialogo(tok, mensajes)
        if len(ids) > contexto or sum(mascara) < 2:
            descartados += 1                                 # no cabe en la ventana del modelo
            continue
        parte = partes["validacion" if hechos % 25 == 0 else "entreno"]      # un 4 % para validar
        parte["tokens"].extend(ids)
        parte["mascara"].extend(mascara)
        parte["limites"].append(len(parte["tokens"]))
        categorias[categoria] = categorias.get(categoria, 0) + 1
        largos += len(ids)
        hechos += 1
        if len(ejemplos) < 14 and sum(1 for e in ejemplos if e["categoria"] == categoria) < 2:
            ejemplos.append({"categoria": categoria, "mensajes": mensajes})
        if hechos >= cuantos:
            break
    if hechos < 200:
        raise ErrorDeProyecto(
            f"Solo se pudieron fabricar {hechos} conversaciones que quepan en {contexto} tokens. "
            "Sube «contexto» en config.json o usa un corpus con textos más cortos."
        )
    proyecto.datos.mkdir(parents=True, exist_ok=True)
    for nombre, parte in partes.items():
        np.savez(proyecto.datos / f"chat_{nombre}.npz",
                 tokens=np.asarray(parte["tokens"], dtype=np.uint16),
                 mascara=np.asarray(parte["mascara"], dtype=np.uint8),
                 limites=np.asarray(parte["limites"], dtype=np.int64))
    manifiesto = {
        "conversaciones": {n: len(p["limites"]) - 1 for n, p in partes.items()},
        "categorias": dict(sorted(categorias.items(), key=lambda par: -par[1])),
        "tokens_por_conversacion": round(largos / hechos, 1),
        "descartadas_por_largas": descartados,
        "segundos": round(time.time() - inicio, 1),
        "ejemplos": ejemplos,
    }
    guardar_json(proyecto.datos / "chat_manifiesto.json", manifiesto)
    resumen = ", ".join(f"{nombre} {miles(cuenta)}" for nombre, cuenta in manifiesto["categorias"].items())
    avisar(f"  Listas {miles(hechos)} conversaciones ({resumen}); media de {manifiesto['tokens_por_conversacion']} tokens.")
    return manifiesto


class LotesChat:
    """Lotes de conversaciones enteras. Cada fila es una conversación desde `<|inicio|>`.

    Las conversaciones tienen largos distintos, así que cada lote se rellena hasta la más larga
    que contiene. Para rellenar poco, se sortean muchas candidatas, se ordenan por largo y se
    toma un grupo de largos parecidos. El objetivo vale -1 (= «no cuenta») en el relleno y en
    todo lo que no dice el asistente.
    """

    def __init__(self, ruta: Path | str, lote: int, semilla: int, dispositivo: str = "cpu") -> None:
        ruta = Path(ruta)
        if not ruta.exists():
            raise ErrorDeProyecto(f"Falta {ruta}. Ejecuta antes:  {ORDEN} afinar   (prepara las conversaciones él solo)")
        with np.load(ruta) as datos:
            self.tokens, self.mascara, self.limites = datos["tokens"], datos["mascara"], datos["limites"]
        self.n = len(self.limites) - 1
        self.lote, self.dispositivo = min(lote, self.n), dispositivo
        self.azar = np.random.default_rng(semilla)

    def siguiente(self) -> tuple[torch.Tensor, torch.Tensor]:
        candidatas = self.azar.integers(0, self.n, size=min(self.n, self.lote * 8))
        largos = self.limites[candidatas + 1] - self.limites[candidatas]
        orden = candidatas[np.argsort(largos, kind="stable")]
        primera = int(self.azar.integers(0, len(orden) - self.lote + 1))
        elegidas = orden[primera : primera + self.lote]
        largo = int((self.limites[elegidas + 1] - self.limites[elegidas]).max())
        x = np.zeros((len(elegidas), largo - 1), dtype=np.int64)
        y = np.full((len(elegidas), largo - 1), -1, dtype=np.int64)
        for fila, i in enumerate(elegidas):
            desde, hasta = self.limites[i], self.limites[i + 1]
            ids, aprender = self.tokens[desde:hasta].astype(np.int64), self.mascara[desde:hasta]
            x[fila, : len(ids) - 1] = ids[:-1]
            y[fila, : len(ids) - 1] = np.where(aprender[1:] == 1, ids[1:], -1)
        return torch.from_numpy(x).to(self.dispositivo), torch.from_numpy(y).to(self.dispositivo)

    def estado(self) -> dict[str, Any]:
        return self.azar.bit_generator.state

    def restaurar(self, estado: dict[str, Any]) -> None:
        self.azar.bit_generator.state = estado


# ========================================================================================
# 3. El ajuste fino
# ========================================================================================
def afinar(
    proyecto: Proyecto,
    dispositivo: str = "auto",
    reanudar: bool = False,
    avisar: Avisar = print,
    progreso: Callable[[int, int, float, float], None] | None = None,
    debe_parar: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Ajusta el modelo base para conversar. Parte de `base/mejor.pt` y deja `chat/mejor.pt`."""
    dispositivo = disp.elegir(dispositivo)
    disp.preparar(dispositivo)
    config = proyecto.config
    cfg = proyecto.config_entreno("chat")
    if not proyecto.mejor("base").exists():
        raise ErrorDeProyecto(f"El proyecto «{proyecto.nombre}» aún no tiene modelo base. Ejecuta antes:  {ORDEN} entrenar")
    if not (proyecto.datos / "chat_entreno.npz").exists():
        preparar_dialogos(proyecto, avisar)

    # Se parte del modelo base ya entrenado: el ajuste es mucho más corto y con una tasa menor,
    # para enseñar el formato sin borrar lo que el modelo ya sabía del idioma.
    modelo, tok, _ = cargar_modelo(proyecto.mejor("base"), dispositivo)
    modelo.train()
    torch.manual_seed(cfg.semilla)
    lotes = LotesChat(proyecto.datos / "chat_entreno.npz", cfg.lote, cfg.semilla, dispositivo)

    def lotes_val() -> LotesChat:
        return LotesChat(proyecto.datos / "chat_validacion.npz", cfg.lote, 4321, dispositivo)

    pregunta = config.get("muestra_chat", "Cuéntame un cuento sobre un gato.")

    def muestra(m: GPT) -> str:
        respuesta = "".join(responder(m, tok, [u(pregunta)], max_nuevos=50, temperatura=0.7, top_k=40, semilla=7))
        return f"{pregunta} → {respuesta}"

    return entrenar(modelo, tok, cfg, lotes, lotes_val, proyecto.etapa("chat"), dispositivo, "chat",
                    reanudar=reanudar, muestra=muestra, avisar=avisar, progreso=progreso, debe_parar=debe_parar,
                    ficha={"proyecto": proyecto.nombre, "receta": config.get("receta"), "arquitectura": config.get("arquitectura")})


# ========================================================================================
# 4. Conversar
# ========================================================================================
def preparar_entrada(tok: TokenizadorBPE, mensajes: list[Mensaje], contexto: int, reserva: int) -> tuple[list[int], int]:
    """Los tokens que verá el modelo para responder, y cuántos mensajes antiguos hubo que olvidar.

    El modelo solo ve `contexto` tokens. Si la conversación no cabe (dejando `reserva` tokens
    para la respuesta), se van quitando los turnos más antiguos: por eso un chatbot «olvida».
    """
    mensajes = list(mensajes)
    olvidados = 0
    while True:
        ids, _ = codificar_dialogo(tok, mensajes, abrir_respuesta=True)
        if len(ids) <= contexto - reserva or len(mensajes) <= 1:
            break
        mensajes, olvidados = mensajes[1:], olvidados + 1
        while mensajes and mensajes[0]["rol"] != "usuario":  # una conversación empieza siempre por el usuario
            mensajes, olvidados = mensajes[1:], olvidados + 1
    if len(ids) > contexto - reserva:                        # un único mensaje demasiado largo: se recorta su final
        sobra = len(ids) - (contexto - reserva)
        ids = ids[: -(sobra + 2)] + ids[-2:]                 # se conservan <|fin|><|asistente|>
    return ids, olvidados


def responder(
    modelo: GPT,
    tok: TokenizadorBPE,
    mensajes: list[Mensaje],
    max_nuevos: int = 200,
    temperatura: float = 0.7,
    top_k: int | None = 40,
    top_p: float | None = None,
    semilla: int | None = None,
) -> Iterator[str]:
    """Va entregando la respuesta del asistente a trozos de texto, según se escribe."""
    contexto = modelo.cfg.contexto
    reserva = min(max_nuevos, contexto // 3)
    ids, _ = preparar_entrada(tok, mensajes, contexto, reserva)
    max_nuevos = max(1, min(max_nuevos, contexto - len(ids)))
    flujo = tok.flujo()
    # Cualquier token especial detiene la respuesta: <|fin|> es el final normal del turno.
    for token in generar(modelo, ids, max_nuevos, temperatura, top_k, top_p, parar=tok.especiales.values(), semilla=semilla):
        trozo = flujo.agregar(token)
        if trozo:
            yield trozo
    resto = flujo.terminar()
    if resto:
        yield resto


def respuesta(modelo: GPT, tok: TokenizadorBPE, mensajes: list[Mensaje], **mandos: Any) -> str:
    """La respuesta completa, de una vez."""
    return "".join(responder(modelo, tok, mensajes, **mandos)).strip()


# ========================================================================================
# 5. Evaluar el comportamiento, no solo la pérdida
# ========================================================================================
def _normalizar(texto: str) -> str:
    return re.sub(r"[\s.¡!¿?,;:]+", " ", texto.lower()).strip()


def evaluar(modelo: GPT, tok: TokenizadorBPE, fuente: str, por_prueba: int = 25, semilla: int = 2024,
            avisar: Avisar | None = None) -> dict[str, dict[str, Any]]:
    """Mide qué hace bien el chatbot y qué no, habilidad por habilidad.

    La pérdida dice cuánto se parece el texto al de entrenamiento; no dice si el modelo obedece.
    Aquí se le hacen preguntas nuevas con respuesta comprobable y se cuenta cuántas acierta.
    Las pruebas «nuevo/nueva/raro» usan nombres o sumas que el modelo no vio NUNCA al entrenar.
    """
    rng = random.Random(semilla)
    resultados: dict[str, dict[str, Any]] = {}
    max_nuevos = min(220, modelo.cfg.contexto // 2)

    def contestar(pregunta: str, limite: int) -> tuple[str, bool]:
        ids, _ = preparar_entrada(tok, [u(pregunta)], modelo.cfg.contexto, min(limite, modelo.cfg.contexto // 3))
        limite = max(1, min(limite, modelo.cfg.contexto - len(ids)))
        nuevos = list(generar(modelo, ids, limite, temperatura=0.0, parar=tok.especiales.values()))
        return tok.decodificar(nuevos).strip(), len(nuevos) < limite       # (texto, ¿se detuvo solo?)

    def probar(nombre: str, descripcion: str, casos: list[tuple[str, Callable[[str], bool], int]]) -> None:
        aciertos, paradas, ejemplo = 0, 0, None
        for pregunta, correcta, limite in casos:
            texto, paro = contestar(pregunta, limite)
            bien = correcta(texto)
            aciertos += bien
            paradas += paro
            if ejemplo is None or (not bien and ejemplo["bien"]):
                ejemplo = {"pregunta": pregunta[-160:], "respuesta": texto[:200], "bien": bool(bien)}
        resultados[nombre] = {"descripcion": descripcion, "aciertos": aciertos, "total": len(casos),
                              "se_detiene": paradas, "ejemplo": ejemplo}
        if avisar:
            avisar(f"  {descripcion:<52} {aciertos:>3}/{len(casos)}")

    def sumas(reservadas: bool) -> list[tuple[str, Callable[[str], bool], int]]:
        pares = [(x, y) for x in range(SUMA_MAXIMA + 1) for y in range(SUMA_MAXIMA + 1) if suma_reservada(x, y) == reservadas]
        return [(f"¿Cuánto es {x} + {y}?", (lambda t, r=x + y: re.search(rf"=\s*{r}\b", t) is not None), 16)
                for x, y in rng.sample(pares, min(por_prueba, len(pares)))]

    if fuente == "sintetico":
        def cuentos(reserva: str | None, con_nombre: bool) -> list[tuple[str, Callable[[str], bool], int]]:
            casos = []
            for _ in range(por_prueba):
                c = cuento(rng, reserva=reserva, variados=False)
                p = c.protagonista
                if con_nombre:
                    pregunta = f"Cuéntame un cuento sobre {p.un_tipo} llamad{p.o} {p.nombre}."
                    casos.append((pregunta, (lambda t, n=p.nombre: n in t), max_nuevos))
                else:
                    pregunta = f"Cuéntame un cuento sobre {p.un_tipo}."
                    casos.append((pregunta, (lambda t, tipo=p.tipo: re.search(rf"\b{tipo}\b", t) is not None), max_nuevos))
            return casos

        def lectura(reserva: str | None) -> list[tuple[str, Callable[[str], bool], int]]:
            casos = []
            for _ in range(por_prueba):
                c = cuento(rng, reserva=reserva, variados=False)
                pregunta, esperada = rng.choice(c.hechos)
                casos.append((f"{c.texto}\n\n{pregunta}", (lambda t, e=esperada: _normalizar(t) == _normalizar(e)), 40))
            return casos

        probar("cuento_personaje", "Cuento sobre el personaje pedido", cuentos(None, False))
        probar("cuento_nombre", "Cuento con el nombre pedido: nombre conocido", cuentos(None, True))
        probar("cuento_nombre_nuevo", "  ...nombre NUNCA visto, de sílabas corrientes", cuentos("nuevos", True))
        probar("cuento_nombre_raro", "  ...nombre NUNCA visto y con letras raras", cuentos("raros", True))
        probar("lectura", "Responder sobre un cuento dado: nombres conocidos", lectura(None))
        probar("lectura_nueva", "  ...con nombres NUNCA vistos", lectura("nuevos"))
    else:
        tipos = ["un gato", "una niña", "un perro", "un conejo", "una princesa", "un oso", "un pájaro", "un niño", "una rana", "un dragón"]
        casos = [(f"Cuéntame un cuento sobre {t}.", (lambda x, palabra=t.split()[1]: palabra in x.lower()), max_nuevos)
                 for t in (rng.choice(tipos) for _ in range(por_prueba))]
        probar("cuento_personaje", "Cuento sobre el personaje pedido", casos)

    probar("suma", "Sumas que estaban en el entrenamiento", sumas(False))
    probar("suma_nueva", "Sumas que NUNCA vio", sumas(True))
    probar("no_se", "Decir «no lo sé» ante preguntas nuevas fuera de su mundo",
           [(p, (lambda t: "no lo sé" in t.lower() or "no sé" in t.lower()), 40) for p in FUERA_DE_ALCANCE_NUEVAS])
    return resultados
