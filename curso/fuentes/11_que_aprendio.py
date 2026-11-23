# %% [markdown]
# # Clase 11 · ¿Qué aprendió de verdad?
#
# El chatbot responde con soltura. Es fácil pensar que «entiende». En esta clase lo ponemos a
# prueba con preguntas **nuevas** cuya respuesta se puede comprobar, y separamos lo que ha
# aprendido de lo que solo ha memorizado.
#
# | | |
# |---|---|
# | **Duración** | 50 minutos |
# | **En el código** | `cero/chat.py`: `evaluar` · `cero/cuentos.py`: nombres reservados |
# | **Al terminar sabrás** | Diseñar una prueba que distinga memorizar de generalizar, e interpretar sus resultados con honestidad |

# %%
import torch

from cero import chat, dispositivo, taller
from cero.chat import u
from cero.cuentos import RESERVAS
from cero.taller import RAPIDO

torch.set_grad_enabled(False)
DISPOSITIVO = dispositivo.elegir("auto")
proyecto = taller.asegurar(taller.proyecto_del_curso(), "chat", DISPOSITIVO)
chatbot, tok, _ = proyecto.cargar("chat", DISPOSITIVO)

# %% [markdown]
# ## 1. Cómo se diseña una prueba justa
#
# Si le preguntamos algo que ya vio al entrenar, un acierto no demuestra nada: puede estar
# recordando. La clave es **reservar** cosas que el modelo no ve nunca:
#
# | Lo reservado | Para qué |
# |---|---|
# | Nombres `NUEVOS` (Ramiro, Selena...): nunca vistos, pero hechos de sílabas corrientes | ¿Sabe **copiar** un nombre que no conoce? |
# | Nombres `RAROS` (Zacarías, Greta...): nunca vistos y con letras poco frecuentes | ¿Hasta dónde llega esa capacidad? |
# | 1 de cada 7 sumas (de 0 a 10) | ¿Sabe **sumar**, o recuerda las sumas que vio? |
# | 10 preguntas «fuera de su mundo» que no estaban en el entrenamiento | ¿Ha aprendido **cuándo** decir «no lo sé»? |
#
# El generador de cuentos y el de conversaciones se aseguran de que nada de eso aparezca jamás
# al entrenar (hay una prueba automática que lo comprueba en 6.000 cuentos).

# %%
print("Nombres NUEVOS:", RESERVAS["nuevos"])
print("Nombres RAROS: ", RESERVAS["raros"])
reservadas = [f"{x}+{y}" for x in range(11) for y in range(11) if chat.suma_reservada(x, y)]
print(f"Sumas reservadas ({len(reservadas)} de 121):", ", ".join(reservadas))

# %% [markdown]
# ## 2. Unos ejemplos a mano
#
# Generamos sin azar (temperatura 0): así el resultado no depende de la suerte del sorteo.

# %%
pruebas = [
    "Cuéntame un cuento sobre una gata llamada Lucía.",       # nombre visto al entrenar
    "Cuéntame un cuento sobre una gata llamada Selena.",      # nombre NUNCA visto (sílabas corrientes)
    "Cuéntame un cuento sobre una gata llamada Greta.",       # nombre NUNCA visto (letras raras)
    "¿Cuánto es 4 + 5?",                                      # suma vista
    "¿Cuánto es 3 + 7?",                                      # suma NUNCA vista (3 + 2·7 = 17, y 17 % 7 = 3)
    "¿Qué idioma se habla en Brasil?",                        # pregunta fuera de su mundo, nueva
]
for pregunta in pruebas:
    print(f"tú   > {pregunta}")
    print(f"cero > {chat.respuesta(chatbot, tok, [u(pregunta)], temperatura=0, max_nuevos=50)[:150]}\n")

# %% [markdown]
# ## 3. La evaluación completa
#
# `chat.evaluar` repite cada tipo de pregunta 25 veces con casos distintos y cuenta aciertos.
# Es la orden `cero evaluar`. Tarda alrededor de un minuto.

# %%
informe = chat.evaluar(chatbot, tok, "sintetico", por_prueba=3 if RAPIDO else 25, avisar=print)

# %% [markdown]
# Cada prueba guarda un ejemplo; si hubo fallos, el ejemplo es un fallo:

# %%
for nombre, r in informe.items():
    e = r["ejemplo"]
    print(f"{'✔' if e['bien'] else '✗'} {nombre:<20} {e['pregunta'][-60:]!r}\n{'':22} -> {e['respuesta'][:100]!r}")

# %% [markdown]
# ## 4. Cómo leer los resultados
#
# Estos son los números que obtuvimos al preparar el curso, entrenando la misma receta varias
# veces en GPU y en CPU (los tuyos caerán en esos márgenes). Interpretarlos es lo importante:
#
# | Prueba | Aciertos | Lo que dice |
# |---|---:|---|
# | Cuento con el personaje pedido | 25/25 | Obedece la petición. |
# | Nombre conocido | 25/25 | Bien... pero podría ser memoria. |
# | Nombre **nunca visto**, sílabas corrientes | 25/25 | **Copia** el nombre de la petición: no lo recuerda, lo trae del contexto con la atención. |
# | Nombre **nunca visto**, letras raras | 0/25 | Lo sustituye por uno conocido parecido («Borja» → «Bruno»). Copiar tokens que casi nunca vio está fuera de su alcance. |
# | Responder sobre un cuento dado | 14-20/25 | Suele acertar; falla sobre todo con datos parecidos entre sí (dónde vivía). |
# | Sumas vistas | 22-24/25 | Las responde... |
# | Sumas **nunca vistas** | 4-8/17 | ...pero no sabe sumar: **memorizó la tabla**. Las nuevas las falla casi siempre por poco («3 + 7 = 9»). |
# | «No lo sé» en preguntas nuevas | 5-10/10 | Reconoce la *forma* de una pregunta que no es de su mundo... casi siempre: es la habilidad que más varía de un entrenamiento a otro. |
#
# Ningún número de esta tabla se ve en la pérdida de validación. Por eso, para un chatbot,
# hay que medir **comportamientos**.

# %% [markdown]
# ## 5. Un experimento: ¿memorizar o copiar?
#
# El corpus del curso mezcla 80 nombres fijos con miles de nombres inventados juntando
# sílabas (`nombre_inventado` en `cero/cuentos.py`). ¿Por qué tanto lío? Hicimos el
# experimento de entrenar el mismo modelo con **solo los 80 nombres fijos**:
#
# | Corpus | Nombre conocido | Nombre nunca visto | Lectura con nombres nunca vistos |
# |---|---:|---:|---:|
# | Solo 80 nombres fijos | 25/25 | **0/25** | **0/25** |
# | Nombres fijos + inventados (el del curso) | 25/25 | **25/25** | **20/25** |
#
# Con 80 nombres, el camino más fácil para bajar la pérdida es **aprendérselos**: el modelo
# no tiene ninguna necesidad de mirar atrás para copiar. Cuando aparece un nombre que no
# conoce, pone uno de los suyos. Con miles de nombres distintos, memorizar deja de compensar
# y el modelo desarrolla otra habilidad: **copiar lo que tiene delante**.
#
# Es una lección que vale para modelos de cualquier tamaño: **lo que un modelo aprende
# depende de los datos**, y la variedad empuja hacia habilidades generales.
#
# Puedes repetirlo. Tarda unos minutos en GPU (bastantes más en CPU), así que está apagado:

# %%
REPETIR_EXPERIMENTO = False          # cámbialo a True para entrenar el modelo de nombres fijos

if REPETIR_EXPERIMENTO or RAPIDO:    # (RAPIDO solo lo usan las pruebas automáticas)
    fijos = taller.proyecto_del_curso("curso-fijos", corpus={"nombres": "fijos"})
    taller.asegurar(fijos, "chat", DISPOSITIVO)
    modelo_fijos, tok_fijos, _ = fijos.cargar("chat", DISPOSITIVO)
    informe_fijos = chat.evaluar(modelo_fijos, tok_fijos, "sintetico", por_prueba=3 if RAPIDO else 25)
    print(f"{'prueba':<22} {'curso':>8} {'fijos':>8}")
    for nombre in informe:
        print(f"{nombre:<22} {informe[nombre]['aciertos']:>5}/{informe[nombre]['total']:<2} {informe_fijos[nombre]['aciertos']:>5}/{informe_fijos[nombre]['total']:<2}")

# %% [markdown]
# ## 6. ¿Entonces, entiende?
#
# Con lo que hemos medido, lo defendible es:
#
# * Predice el siguiente token a partir de regularidades de sus datos, y esas regularidades
#   incluyen **mecanismos generales** (copiar un nombre del contexto, mantener la concordancia).
# * Otras cosas que parecen razonamiento son **memoria** (las sumas).
# * Todo lo que no estaba en su mundo, no lo sabe. Y si no le hubiéramos enseñado a decir
#   «no lo sé», se lo inventaría con la misma soltura.
#
# Afirmar más que eso («entiende», «razona», «sabe») exige pruebas que no hemos hecho.
#
# ## 7. ¿Qué le falta para ser un asistente como los de verdad?
#
# | Lo que tienen los asistentes reales | Nosotros |
# |---|---|
# | Billones de tokens de texto variado (web, libros, código) | 4 millones de tokens de cuentos |
# | Miles de millones de parámetros | 0,9 millones |
# | Conversaciones escritas y revisadas por personas | Conversaciones fabricadas con reglas |
# | **Alineamiento con preferencias** (RLHF, DPO): aprender qué respuesta prefiere una persona entre dos | No lo hacemos |
# | **Aprendizaje por refuerzo** con respuestas comprobables (matemáticas, código) | No lo hacemos |
# | Herramientas, búsqueda, memoria larga | Solo la ventana de 256 tokens |
#
# Las dos etapas que faltan (preferencias y refuerzo) parten de un modelo como el nuestro y
# cambian solo los datos y la función de pérdida. El repositorio *train-llm-from-scratch*, en
# el que se inspira este proyecto, las implementa en PyTorch si quieres seguir por ahí.
#
# ## Resumen
#
# * Para saber si generaliza, pregúntale por cosas que **nunca** vio.
# * Nuestro chatbot copia nombres nuevos, obedece y sabe decir «no lo sé»; no sabe sumar.
# * La variedad de los datos decide si memoriza o desarrolla una habilidad general.
#
# ## Ejercicios
#
# 1. Pide cuentos con tu nombre y los de tus compañeros. ¿Cuáles copia bien? ¿Tienen algo en
#    común los que fallan? (Pista: tokenízalos como en la clase 1.)
# 2. Inventa tres preguntas «fuera de su mundo» con otra forma («Dime cuál es...», «Explica...»).
#    ¿Sigue diciendo «no lo sé»? ¿Y si la pregunta es sobre un gato pero fuera de su mundo
#    («¿cuánto pesa un gato?»)?
# 3. Pon `REPETIR_EXPERIMENTO = True` y comprueba la tabla de la sección 5.
# 4. Propón una forma de enseñarle a **sumar** de verdad en lugar de memorizar. (Idea: ¿y si
#    la respuesta incluyera los pasos intermedios?) ¿Qué habría que cambiar en `cero/chat.py`?
# 5. Escribe una pregunta nueva para `evaluar` con su comprobación. ¿Qué habilidad mide?
#
# **Siguiente:** [Clase 12 · Experimentos y escala](12_experimentos.ipynb)
