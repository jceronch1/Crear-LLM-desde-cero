"""Un mundo de cuentos diminuto, generado por reglas. No descarga nada.

¿Por qué inventar los datos? Porque un modelo de un millón de parámetros no puede aprender
«el español», pero SÍ puede aprender un mundo pequeño y regular en pocos minutos de CPU.
Este archivo fabrica ese mundo: cuentos cortos con un vocabulario reducido (animales,
nombres, lugares, objetos) y ocho tramas distintas. Es la misma idea que hay detrás de
TinyStories (Eldan y Li, 2023), llevada al extremo para que quepa en una clase.

Tiene dos ventajas pedagógicas más:

* Como cada cuento sale de una ficha de datos (quién, dónde, qué pasó), sabemos las
  respuestas correctas a preguntas sobre él. Con eso se fabrican las conversaciones que
  convierten al modelo en un chatbot (cero/chat.py) y se mide si de verdad «entiende».
* El español obliga a concordar («una gata pequeña», «un gato pequeño»). Ver al modelo
  aprender la concordancia es una forma muy clara de ver que aprende gramática.

Hay nombres reservados (`NOMBRES_NUEVOS_*`) que NUNCA aparecen en los datos de
entrenamiento: sirven para comprobar si el modelo sabe copiar un nombre que no ha visto.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# El vocabulario del mundo
# --------------------------------------------------------------------------------------
TIPOS: list[tuple[str, str]] = [
    ("niño", "m"), ("gato", "m"), ("perro", "m"), ("conejo", "m"), ("ratón", "m"), ("oso", "m"),
    ("zorro", "m"), ("pájaro", "m"), ("dragón", "m"), ("robot", "m"), ("pato", "m"), ("león", "m"),
    ("lobo", "m"), ("búho", "m"), ("caballo", "m"), ("elefante", "m"), ("mono", "m"), ("pingüino", "m"),
    ("erizo", "m"), ("burro", "m"),
    ("niña", "f"), ("gata", "f"), ("tortuga", "f"), ("rana", "f"), ("abeja", "f"), ("ardilla", "f"),
    ("oveja", "f"), ("mariposa", "f"), ("vaca", "f"), ("hormiga", "f"), ("princesa", "f"), ("osa", "f"),
    ("jirafa", "f"), ("ballena", "f"), ("gallina", "f"), ("cabra", "f"), ("luciérnaga", "f"),
    ("coneja", "f"), ("leona", "f"), ("foca", "f"),
]

NOMBRES_M = [
    "Leo", "Hugo", "Mateo", "Lucas", "Martín", "Daniel", "Pablo", "Diego", "Bruno", "Tomás", "Nico",
    "Samuel", "Iván", "Marcos", "Andrés", "Javier", "Carlos", "Simón", "Gael", "Darío", "Luis", "Pedro",
    "Juan", "Omar", "Raúl", "Adrián", "Felipe", "Emilio", "Toby", "Milo", "Coco", "Rufo", "Tito", "Max",
    "Pipo", "Óscar", "Ángel", "Rubén", "Jorge", "Sergio",
]
NOMBRES_F = [
    "Ana", "Lucía", "Sofía", "Martina", "Paula", "Julia", "Emma", "Daniela", "Carla", "Alba", "Sara",
    "Noa", "Vera", "Elena", "Irene", "Laura", "Marta", "Nora", "Olivia", "Carmen", "Rosa", "Inés",
    "Clara", "Lola", "Luna", "Nube", "Mimi", "Lili", "Kira", "Nala", "Perla", "Gala", "Abril", "Jimena",
    "Aitana", "Valeria", "Camila", "Renata", "Elsa", "Alma",
]
# Reservados para la evaluación: el modelo no los ve jamás mientras aprende. Dos grados de novedad:
#   NUEVOS  nombres nunca vistos, pero formados con sílabas corrientes en este mundo
#   RAROS   nombres nunca vistos y, además, con letras que apenas salen en los cuentos (Z, Q, G...)
NOMBRES_NUEVOS_M = ["Ramiro", "Remo", "Lino", "Nilo", "Dino", "Marino"]
NOMBRES_NUEVOS_F = ["Selena", "Malena", "Lorena", "Romina", "Milena", "Samira"]
NOMBRES_RAROS_M = ["Ulises", "Fermín", "Teo", "Borja", "Quique", "Zacarías"]
NOMBRES_RAROS_F = ["Greta", "Maite", "Olga", "Zoe", "Petra", "Yara"]
RESERVAS = {"nuevos": (NOMBRES_NUEVOS_M, NOMBRES_NUEVOS_F), "raros": (NOMBRES_RAROS_M, NOMBRES_RAROS_F)}

# Además de los nombres de las listas, se pueden INVENTAR nombres juntando sílabas («Lira»,
# «Dano», «Sumeli»...). Hay decenas de miles posibles. ¿Para qué? Con 80 nombres fijos el
# modelo puede aprendérselos de memoria; con miles distintos no le queda más remedio que
# aprender a COPIAR el nombre que aparece en el texto. Es el experimento de la clase 11.
_CONSONANTES, _VOCALES = "dlmnrsv", "aeiou"
_NO_COMO_NOMBRE = {"rana", "mono", "mona", "lelo", "lela", "mala", "malo", "nada", "sida", "nena", "nene", "loro",
                   "dedo", "duro", "dura", "mama", "sosa", "soso", "muda", "mudo", "rara", "raro", "seno", "mero"}


# «Ramiro» o «Selena» se podrían formar juntando sílabas: se impide, para que sigan siendo nunca vistos.
_RESERVADOS = set(NOMBRES_NUEVOS_M + NOMBRES_NUEVOS_F + NOMBRES_RAROS_M + NOMBRES_RAROS_F)


def nombre_inventado(rng: random.Random) -> str:
    while True:
        silabas = rng.choice([2, 2, 3])
        nombre = "".join(rng.choice(_CONSONANTES) + rng.choice(_VOCALES) for _ in range(silabas))
        if nombre not in _NO_COMO_NOMBRE and nombre.capitalize() not in _RESERVADOS:
            return nombre.capitalize()


ADJETIVOS = [  # (masculino, femenino)
    ("pequeño", "pequeña"), ("valiente", "valiente"), ("curioso", "curiosa"), ("alegre", "alegre"),
    ("tímido", "tímida"), ("amable", "amable"), ("travieso", "traviesa"), ("listo", "lista"),
    ("tranquilo", "tranquila"), ("juguetón", "juguetona"), ("soñador", "soñadora"), ("generoso", "generosa"),
]
LUGARES = [
    "en el bosque", "en la playa", "cerca del río", "en la montaña", "en una granja", "en un jardín",
    "en un pueblo pequeño", "junto al lago", "en un castillo", "en la selva", "en una isla",
    "en el campo", "en una ciudad grande", "en un valle verde",
]
OBJETOS = [  # (como se presenta, como se nombra después)
    ("una pelota roja", "la pelota"), ("un libro de cuentos", "el libro"), ("una llave dorada", "la llave"),
    ("un sombrero azul", "el sombrero"), ("una cometa de colores", "la cometa"), ("un tambor pequeño", "el tambor"),
    ("una bufanda verde", "la bufanda"), ("un barco de papel", "el barco"), ("una linterna", "la linterna"),
    ("un collar de flores", "el collar"), ("una manta suave", "la manta"), ("un paraguas amarillo", "el paraguas"),
    ("una caja de música", "la caja"), ("un globo rojo", "el globo"), ("una flauta", "la flauta"),
    ("un cubo azul", "el cubo"),
]
ESCONDITES = [
    "debajo de un árbol", "detrás de una roca", "dentro de un tronco", "entre las flores",
    "junto a un arbusto", "debajo de un puente", "al lado de una fuente", "encima de una piedra",
]
ACTIVIDADES = [
    "jugar a la pelota", "cantar canciones", "bailar", "pintar dibujos", "nadar", "correr por el campo",
    "leer cuentos", "saltar charcos", "volar cometas", "buscar tesoros", "mirar las estrellas",
    "construir castillos de arena", "plantar flores", "contar historias", "trepar a los árboles",
    "hacer galletas",
]
MIEDOS = ["la oscuridad", "las tormentas", "el agua", "las alturas", "los ruidos fuertes", "el viento"]
HABILIDADES = [
    "nadar", "volar", "cantar", "pintar", "silbar", "saltar muy alto", "montar en bicicleta",
    "tocar el tambor", "contar hasta diez", "leer",
]
COMIDAS = [
    ("una manzana", "la manzana"), ("un trozo de pastel", "el pastel"), ("unas galletas", "las galletas"),
    ("un pan caliente", "el pan"), ("unas fresas", "las fresas"), ("una sandía", "la sandía"),
    ("un poco de miel", "la miel"), ("unas nueces", "las nueces"),
]
DESTINOS = ["una cueva", "una isla pequeña", "un árbol muy alto", "una cascada", "un faro", "una colina"]
TESOROS = [
    "monedas brillantes", "piedras de colores", "cartas muy viejas", "juguetes de madera",
    "semillas mágicas", "caramelos", "libros con dibujos", "conchas del mar",
]
APUROS = [  # (qué le pasaba a quien pide ayuda, cómo se le ayuda)
    ("se había caído en un agujero", "le tendió una rama y tiró con fuerza"),
    ("no podía alcanzar una fruta del árbol", "se subió a una piedra y le bajó la fruta"),
    ("se había caído y no podía levantarse", "le ayudó a levantarse despacio"),
    ("había perdido el camino a casa", "caminó a su lado hasta encontrar el camino"),
    ("tenía mucho frío", "le prestó su manta"),
    ("tenía mucha hambre", "compartió su comida"),
]
MORALES = [
    "compartir nos hace felices", "pedir ayuda no es malo", "los amigos se cuidan",
    "con paciencia todo sale mejor", "ser amable siempre vale la pena", "juntos todo es más fácil",
]
TRAMAS = ["perdido", "amigo", "miedo", "compartir", "tesoro", "ayudar", "aprender", "lluvia"]


# --------------------------------------------------------------------------------------
# Personajes y concordancia
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Personaje:
    tipo: str        # "gata"
    genero: str      # "m" o "f"
    nombre: str      # "Luna"
    adjetivo: str    # ya concordado: "curiosa"

    @property
    def o(self) -> str:
        """La terminación que concuerda: «content-o» / «content-a»."""
        return "o" if self.genero == "m" else "a"

    @property
    def un_tipo(self) -> str:
        return f"{'un' if self.genero == 'm' else 'una'} {self.tipo}"          # «una gata»

    @property
    def un(self) -> str:
        return f"{self.un_tipo} {self.adjetivo}"                              # «una gata curiosa»

    @property
    def el(self) -> str:
        return f"{'el' if self.genero == 'm' else 'la'} {self.tipo}"          # «la gata»

    @property
    def pronombre(self) -> str:
        return "él" if self.genero == "m" else "ella"


@dataclass
class Cuento:
    texto: str
    trama: str
    protagonista: Personaje
    amigo: Personaje | None
    hechos: list[tuple[str, str]] = field(default_factory=list)   # (pregunta, respuesta) sobre el cuento


def _personaje(rng: random.Random, tipo: str | None = None, nombre: str | None = None,
               evitar: Personaje | None = None, reserva: str | None = None, variados: bool = True) -> Personaje:
    """Sortea un personaje. `evitar` impide repetir el tipo o el nombre del protagonista.

    `variados=True`: la mitad de las veces el nombre es inventado.
    `reserva="nuevos"` o `"raros"`: el nombre sale de una de las listas reservadas para evaluar.
    """
    while True:
        if tipo is not None:
            genero = dict(TIPOS).get(tipo)
            if genero is None:
                raise ValueError(f"No conozco el personaje «{tipo}». Los que hay: {', '.join(t for t, _ in TIPOS)}")
            elegido = (tipo, genero)
        else:
            elegido = rng.choice(TIPOS)
        genero = elegido[1]
        if nombre is not None:
            n = nombre
        elif reserva is not None:
            n = rng.choice(RESERVAS[reserva][0 if genero == "m" else 1])
        elif variados and rng.random() < 0.5:
            n = nombre_inventado(rng)
        else:
            n = rng.choice(NOMBRES_M if genero == "m" else NOMBRES_F)
        adjetivo = rng.choice(ADJETIVOS)[0 if genero == "m" else 1]
        p = Personaje(elegido[0], genero, n, adjetivo)
        if evitar is None or (p.tipo != evitar.tipo and p.nombre != evitar.nombre):
            return p


def _plural(n: Personaje, a: Personaje) -> dict[str, str]:
    """Formas en plural para la pareja: solo son femeninas si las dos lo son."""
    f = n.genero == "f" and a.genero == "f"
    return {"juntos": "juntas" if f else "juntos", "amigos": "amigas" if f else "amigos",
            "contentos": "contentas" if f else "contentos", "los_dos": "las dos" if f else "los dos"}


def _su(definido: str) -> str:
    """«la pelota» -> «pelota», para decir «su pelota» o «mi pelota»."""
    return definido.split(" ", 1)[1]


# --------------------------------------------------------------------------------------
# Las ocho tramas. Cada una devuelve (oraciones, hechos).
# --------------------------------------------------------------------------------------
def _perdido(rng, n: Personaje, a: Personaje, pl):
    indef, defi = rng.choice(OBJETOS)
    lo = "lo" if indef.startswith("un ") else "la"
    escondite = rng.choice(ESCONDITES)
    cosa = _su(defi)
    o = [
        rng.choice([f"Un día, {n.nombre} perdió su {cosa}.", f"Una mañana, {n.nombre} no encontraba su {cosa}.",
                    f"{n.nombre} tenía {indef}, pero un día {lo} perdió."]),
        f"Buscó por todas partes, pero no {lo} encontró.",
        rng.choice([f"{n.nombre} se puso muy triste.", f"Estaba tan triste que se sentó a llorar."]),
        rng.choice([f"Entonces apareció {a.nombre}, {a.un}.", f"Por allí pasaba {a.nombre}, {a.un}."]),
        f"—¿Qué te pasa? —preguntó {a.nombre}.",
        f"—He perdido mi {cosa} —dijo {n.nombre}.",
        rng.choice([f"—Yo te ayudo a buscar —dijo {a.nombre}.", f"—No llores, vamos a buscar {pl['juntos']} —dijo {a.nombre}."]),
        rng.choice([f"Buscaron {pl['juntos']} y por fin encontraron {defi} {escondite}.",
                    f"Al final, {a.nombre} vio {defi} {escondite}."]),
        rng.choice([f"{n.nombre} se puso muy content{n.o}.", f"—¡Gracias, {a.nombre}! —dijo {n.nombre} con una gran sonrisa."]),
    ]
    h = [(f"¿Qué perdió {n.nombre}?", f"{n.nombre} perdió su {cosa}."),
         (f"¿Quién ayudó a {n.nombre}?", f"{a.nombre} ayudó a {n.nombre}."),
         (f"¿Dónde estaba {defi}?", f"{defi.capitalize()} estaba {escondite}.")]
    return o, h


def _amigo(rng, n: Personaje, a: Personaje, pl):
    actividad = rng.choice(ACTIVIDADES)
    o = [
        rng.choice([f"{n.nombre} no tenía con quién jugar y a veces se sentía sol{n.o}.",
                    f"Pero {n.nombre} no tenía amigos y se aburría mucho."]),
        rng.choice([f"Un día conoció a {a.nombre}, {a.un}.", f"Una tarde se encontró con {a.nombre}, {a.un}."]),
        f"—Hola, me llamo {n.nombre}. ¿Quieres jugar conmigo? —preguntó.",
        rng.choice([f"—¡Claro que sí! —respondió {a.nombre}.", f"—Sí, me encantaría —dijo {a.nombre}."]),
        rng.choice([f"Se pusieron a {actividad} y se rieron mucho.", f"Empezaron a {actividad} y el tiempo pasó volando."]),
        rng.choice([f"Desde ese día, {n.nombre} y {a.nombre} fueron grandes {pl['amigos']}.",
                    f"{n.nombre} ya nunca se sintió sol{n.o}."]),
    ]
    h = [(f"¿A quién conoció {n.nombre}?", f"{n.nombre} conoció a {a.nombre}."),
         (f"¿Quién es {a.nombre}?", f"{a.nombre} es {a.un}.")]
    return o, h


def _miedo(rng, n: Personaje, a: Personaje, pl):
    miedo = rng.choice(MIEDOS)
    o = [
        f"{n.nombre} tenía miedo de {miedo}.",
        rng.choice([f"Cada vez que pensaba en eso, se ponía a temblar.", f"Por eso casi nunca salía de casa."]),
        rng.choice([f"Un día, {a.nombre}, {a.un}, fue a verl{n.o}.", f"Una tarde llegó {a.nombre}, {a.un}."]),
        f"—No tengas miedo, yo estoy contigo —dijo {a.nombre}.",
        rng.choice([f"{n.nombre} respiró hondo y lo intentó.", f"{n.nombre} cerró los ojos y dio un paso."]),
        "Poco a poco, el miedo se hizo pequeño.",
        rng.choice([f"Desde entonces, {n.nombre} ya no tuvo miedo de {miedo}.",
                    f"{n.nombre} descubrió que era muy valiente."]),
    ]
    h = [(f"¿De qué tenía miedo {n.nombre}?", f"{n.nombre} tenía miedo de {miedo}."),
         (f"¿Quién ayudó a {n.nombre}?", f"{a.nombre} ayudó a {n.nombre}.")]
    return o, h


def _compartir(rng, n: Personaje, a: Personaje, pl):
    indef, defi = rng.choice(COMIDAS)
    o = [
        rng.choice([f"Un día, {n.nombre} encontró {indef}.", f"Una tarde, {n.nombre} tenía {indef} para merendar."]),
        f"Cerca de allí estaba {a.nombre}, {a.un}, que tenía mucha hambre.",
        rng.choice([f"{n.nombre} pensó un momento.", f"Al principio, {n.nombre} quería {defi} solo para {n.pronombre}."]),
        f"Después compartió {defi} con {a.nombre}.",
        f"—¡Muchas gracias! —dijo {a.nombre}.",
        f"Comieron {pl['juntos']} y estaban muy {pl['contentos']}.",
        f"{n.nombre} aprendió que compartir nos hace felices.",
    ]
    h = [(f"¿Qué compartió {n.nombre}?", f"{n.nombre} compartió {defi}."),
         (f"¿Con quién compartió {n.nombre}?", f"{n.nombre} compartió con {a.nombre}.")]
    return o, h


def _tesoro(rng, n: Personaje, a: Personaje, pl):
    destino, tesoro, escondite = rng.choice(DESTINOS), rng.choice(TESOROS), rng.choice(ESCONDITES)
    o = [
        f"Un día, {n.nombre} encontró un mapa viejo {escondite}.",
        f"El mapa mostraba un camino hasta {destino}.",
        rng.choice([f"{n.nombre} caminó y caminó hasta llegar.", f"Fue un viaje largo, pero {n.nombre} no se rindió."]),
        f"Allí había un cofre lleno de {tesoro}.",
        rng.choice([f"{n.nombre} volvió a casa muy content{n.o} y se lo contó a todos.",
                    f"Guardó el tesoro y volvió a casa antes de la noche."]),
        rng.choice(["Fue la mejor aventura de su vida.", f"Esa noche, {n.nombre} durmió con una sonrisa."]),
    ]
    h = [(f"¿Qué había en el cofre?", f"En el cofre había {tesoro}."),
         (f"¿Qué encontró {n.nombre}?", f"{n.nombre} encontró un mapa viejo."),
         (f"¿Hasta dónde llevaba el mapa?", f"El mapa llevaba hasta {destino}.")]
    return o, h


def _ayudar(rng, n: Personaje, a: Personaje, pl):
    apuro, ayuda = rng.choice(APUROS)
    moral = rng.choice(MORALES)
    o = [
        rng.choice([f"Un día, mientras paseaba, {n.nombre} oyó un ruido.", f"Una mañana, {n.nombre} escuchó que alguien pedía ayuda."]),
        f"Era {a.nombre}, {a.un}, que {apuro}.",
        f"{n.nombre} no lo dudó: {ayuda}.",
        f"—Gracias por ayudarme —dijo {a.nombre}.",
        f"—Para eso están los amigos —respondió {n.nombre}.",
        f"Ese día, {n.nombre} aprendió que {moral}.",
    ]
    h = [(f"¿A quién ayudó {n.nombre}?", f"{n.nombre} ayudó a {a.nombre}."),
         (f"¿Quién es {a.nombre}?", f"{a.nombre} es {a.un}.")]
    return o, h


def _aprender(rng, n: Personaje, a: Personaje, pl):
    habilidad = rng.choice(HABILIDADES)
    o = [
        f"{n.nombre} quería aprender a {habilidad}.",
        rng.choice(["Lo intentó una vez, pero no le salió.", "Al principio le costaba mucho."]),
        f"—No te rindas —le dijo {a.nombre}, {a.un}—. Hay que practicar.",
        rng.choice([f"{n.nombre} practicó todos los días.", f"Cada mañana, {n.nombre} lo intentaba otra vez."]),
        "Por fin, un día lo consiguió.",
        f"{n.nombre} estaba muy orgullos{n.o}.",
        rng.choice([f"Aprendió que con paciencia todo sale mejor.", f"{a.nombre} aplaudió con alegría."]),
    ]
    h = [(f"¿Qué quería aprender {n.nombre}?", f"{n.nombre} quería aprender a {habilidad}."),
         (f"¿Quién animó a {n.nombre}?", f"{a.nombre} animó a {n.nombre}.")]
    return o, h


def _lluvia(rng, n: Personaje, a: Personaje, pl):
    actividad = rng.choice(ACTIVIDADES)
    o = [
        f"Un día, {n.nombre} quería salir a {actividad}, pero empezó a llover.",
        f"{n.nombre} miró por la ventana y se puso triste.",
        f"Entonces llegó {a.nombre}, {a.un}, con una idea.",
        rng.choice([f"—Podemos jugar dentro de casa —dijo {a.nombre}.", f"—Vamos a saltar en los charcos —dijo {a.nombre}."]),
        f"Jugaron {pl['juntos']} toda la tarde y se divirtieron mucho.",
        rng.choice(["Cuando salió el sol, vieron un arcoíris.", f"Al final, {pl['los_dos']} estaban muy {pl['contentos']}."]),
    ]
    h = [(f"¿Qué quería hacer {n.nombre}?", f"{n.nombre} quería salir a {actividad}."),
         (f"¿Quién llegó con una idea?", f"{a.nombre} llegó con una idea.")]
    return o, h


_TRAMAS = {"perdido": _perdido, "amigo": _amigo, "miedo": _miedo, "compartir": _compartir,
           "tesoro": _tesoro, "ayudar": _ayudar, "aprender": _aprender, "lluvia": _lluvia}


# --------------------------------------------------------------------------------------
# Un cuento entero
# --------------------------------------------------------------------------------------
def cuento(rng: random.Random, tipo: str | None = None, nombre: str | None = None,
           trama: str | None = None, reserva: str | None = None, variados: bool = True) -> Cuento:
    """Escribe un cuento. Se puede fijar el tipo de protagonista, su nombre y la trama.

    `variados=False` usa solo los 80 nombres de las listas. `reserva="nuevos"` o `"raros"` usa
    nombres reservados, que nunca aparecen al entrenar (solo para evaluar).
    """
    n = _personaje(rng, tipo, nombre, reserva=reserva, variados=variados)
    a = _personaje(rng, evitar=n, reserva=reserva, variados=variados)
    pl = _plural(n, a)
    lugar = rng.choice(LUGARES)
    trama = trama or rng.choice(TRAMAS)

    # Presentación: quién es y dónde vive.
    entrada = rng.randrange(4)
    if entrada == 0:
        oraciones = [f"Había una vez {n.un} que se llamaba {n.nombre}.", f"{n.nombre} vivía {lugar}."]
    elif entrada == 1:
        oraciones = [f"Érase una vez {n.un} llamad{n.o} {n.nombre}.", f"Vivía {lugar} con su familia."]
    elif entrada == 2:
        oraciones = [f"{n.nombre} era {n.un} que vivía {lugar}."]
    else:
        oraciones = [f"{lugar.capitalize()} vivía {n.un} llamad{n.o} {n.nombre}."]
    hechos = [(f"¿Cómo se llama {n.el}?", f"{n.el.capitalize()} se llama {n.nombre}."),
              (f"¿Dónde vivía {n.nombre}?", f"{n.nombre} vivía {lugar}.")]

    # A veces, un gusto del protagonista.
    if rng.random() < 0.5:
        gusto = rng.choice(ACTIVIDADES)
        oraciones.append(rng.choice([f"A {n.nombre} le gustaba mucho {gusto}.", f"Lo que más le gustaba era {gusto}."]))
        hechos.append((f"¿Qué le gustaba a {n.nombre}?", f"A {n.nombre} le gustaba {gusto}."))

    cuerpo, mas_hechos = _TRAMAS[trama](rng, n, a, pl)
    amigo = None if trama == "tesoro" else a                # en la trama del tesoro no interviene nadie más
    return Cuento(" ".join(oraciones + cuerpo), trama, n, amigo, hechos + mas_hechos)


def generar_cuentos(cuantos: int, semilla: int = 0, variados: bool = True) -> Iterator[str]:
    """Entrega `cuantos` cuentos distintos. Con la misma semilla salen siempre los mismos."""
    rng = random.Random(semilla)
    for _ in range(cuantos):
        yield cuento(rng, variados=variados).texto
