# %% [markdown]
# # Clase 2 · Los datos: qué se le enseña exactamente al modelo
#
# Ya sabemos convertir texto en números. Ahora toca decidir **qué texto** usamos, cómo lo
# repartimos y, sobre todo, **cuál es el ejercicio** que el modelo va a repetir millones de veces.
#
# | | |
# |---|---|
# | **Duración** | 40 minutos |
# | **En el artículo** | Figura 1 («outputs shifted right») y §5.1 |
# | **En el código** | `cero/datos.py`, `cero/cuentos.py` |
# | **Al terminar sabrás** | Qué son x e y, por qué se aparta una validación y cómo se forma un lote |

# %% [markdown]
# ## 1. El corpus
#
# Un modelo de un millón de parámetros no puede aprender «el español». Pero sí puede aprender
# un **mundo pequeño y regular** en pocos minutos. Nuestro primer corpus son cuentos generados
# por reglas (`cero/cuentos.py`): pocos personajes, pocos lugares, ocho tramas. No hay que
# descargar nada y funcionan igual en cualquier ordenador.
#
# (Es la misma idea que *TinyStories*, un corpus de cuentos sencillos creado justamente para
# estudiar modelos pequeños. En la clase 12 entrenaremos con cuentos reales.)

# %%
from cero.cuentos import generar_cuentos
from cero.taller import RAPIDO

cuentos = list(generar_cuentos(500 if RAPIDO else 5000, semilla=0))
print(f"{len(cuentos):,} cuentos, {sum(len(c) for c in cuentos):,} caracteres en total\n")
for cuento in cuentos[:2]:
    print(cuento, "\n")

# %% [markdown]
# Fíjate en que los adjetivos y los artículos **concuerdan** con el personaje («una gata
# curiosa», «un perro travieso») y en que cada cuento repite el nombre del protagonista
# varias veces. Dentro de unas clases comprobaremos si el modelo aprende esas dos cosas.

# %% [markdown]
# ## 2. Limpiar
#
# Los textos reales llegan sucios: espacios dobles, saltos de línea de Windows, la misma letra
# con tilde escrita de dos formas distintas... Antes de tokenizar se dejan en una forma única.
# Si no, el modelo gastaría capacidad en aprender que «niña» y «niña» son lo mismo.

# %%
from cero import datos

sucio = "Había   una vez\tun gato.\r\n\r\n\r\n\r\nSe llamaba  Milo. "
print(repr(sucio))
print(repr(datos.limpiar(sucio)))

# %% [markdown]
# ## 3. Entrenamiento y validación
#
# Imagina que estudias un examen memorizando las respuestas de los exámenes anteriores. Si te
# vuelven a poner **las mismas** preguntas sacarás un diez... sin haber aprendido nada.
#
# A un modelo le pasa igual. Por eso se **aparta una parte del corpus** que el modelo no ve
# nunca mientras aprende: la *validación*. Medir ahí es la única forma de saber si ha
# aprendido a escribir cuentos o solo se ha aprendido de memoria los que vio.
#
# ¿Cómo se decide qué cuento va a cada lado? Con una **huella** del contenido: un número que
# sale del propio texto. Así el reparto no depende del orden ni del azar, y el mismo cuento
# cae siempre en el mismo lado aunque el corpus cambie.

# %%
def va_a_validacion(texto, porcentaje=2):
    return datos.huella(texto) % 100 < porcentaje


entreno = [c for c in cuentos if not va_a_validacion(c)]
validacion = [c for c in cuentos if va_a_validacion(c)]
print(f"entrenamiento: {len(entreno):,} cuentos · validación: {len(validacion):,} cuentos ({100 * len(validacion) / len(cuentos):.1f} %)")
print("¿Algún cuento en los dos lados?", bool(set(entreno) & set(validacion)))

# %% [markdown]
# > **Regla de oro.** La validación no se toca para nada que sea «aprender»: ni para entrenar
# > el modelo, ni para entrenar el tokenizador. Por eso el tokenizador de la clase 1 se
# > entrena solo con la parte de entrenamiento.

# %% [markdown]
# ## 4. El ejercicio: adivinar el token siguiente
#
# Aquí está la idea más importante del curso. No hace falta que nadie etiquete nada: **la
# respuesta correcta ya está en el texto**, un lugar más a la derecha.

# %%
import torch

from cero.tokenizador import INICIO, TokenizadorBPE

tok = TokenizadorBPE.entrenar(entreno, vocab=1024)
ids = [tok.id(INICIO)] + tok.codificar_texto(cuentos[0])
print("El modelo ve...".ljust(62), "y debe predecir")
print("-" * 80)
for i in range(1, 9):
    visto = "".join(tok.texto_de(t) for t in ids[:i])
    print(f"{visto[-58:]!r:62} {tok.texto_de(ids[i])!r}")

# %% [markdown]
# Cada posición del texto es un ejercicio: «con todo lo anterior, ¿qué viene ahora?».
# Esto se escribe con dos listas del mismo largo:
#
# * **x**, la entrada: los tokens de una ventana del texto.
# * **y**, el objetivo: **los mismos tokens, desplazados un lugar**.
#
# En la Figura 1 del artículo es la etiqueta *outputs (shifted right)* debajo del decodificador.

# %%
ventana = ids[:9]
x, y = ventana[:-1], ventana[1:]
print("x:", [tok.texto_de(t) for t in x])
print("y:", [tok.texto_de(t) for t in y])

# %% [markdown]
# ### Tu turno · construir un lote
#
# No se entrena de ventana en ventana, sino con varias a la vez: un **lote** (*batch*).
# Completa la función: recibe todos los tokens del corpus (un tensor de una dimensión), una
# lista con la posición donde empieza cada ventana y el tamaño de la ventana (`contexto`).
# Debe devolver `x` e `y`, cada uno de forma `(número de ventanas, contexto)`.

# %%
from cero import taller


def hacer_lote(tokens, inicios, contexto):
    # TU CÓDIGO: para cada inicio i, la fila de x es tokens[i : i + contexto]
    # y la de y es la misma ventana empezando un token después.
    # Pista: torch.stack([...]) apila una lista de tensores en filas.
    raise NotImplementedError


taller.comprobar_lote(hacer_lote)

# %% [markdown]
# ## 5. La etapa de datos del proyecto
#
# `cero datos` (o `datos.preparar`) hace todo lo de esta clase de una vez: lee el corpus, lo
# limpia, aparta la validación, entrena el tokenizador con el resto y guarda los tokens en
# dos archivos. Lo hacemos para el proyecto del curso:

# %%
proyecto = taller.proyecto_del_curso()
taller.asegurar(proyecto, "datos")
manifiesto = proyecto.estado()["datos"]
for parte, cuenta in manifiesto["partes"].items():
    print(f"{parte:<11} {cuenta['documentos']:>7,} cuentos  {cuenta['tokens']:>10,} tokens")
print(f"\n{manifiesto['caracteres_por_token']} caracteres por token · vocabulario de {manifiesto['vocab']} tokens")
print("Archivos:", sorted(ruta.name for ruta in proyecto.datos.iterdir()))

# %% [markdown]
# Los tokens se guardan como números de 16 bits, uno detrás de otro. `Lotes` saca de ahí
# ventanas al azar, igual que tu función:

# %%
contexto = proyecto.config["modelo"]["contexto"]
lotes = datos.Lotes(proyecto.datos / "entreno.bin", contexto=contexto, lote=4, semilla=0)
x, y = lotes.siguiente()
print("forma de x:", tuple(x.shape), "· forma de y:", tuple(y.shape))
tok = proyecto.tokenizador()
print("\nx[0] empieza así:", repr(tok.decodificar(x[0, :25].tolist())))
print("y[0] empieza así:", repr(tok.decodificar(y[0, :25].tolist())))

# %% [markdown]
# Las ventanas empiezan en cualquier sitio, también a mitad de un cuento o de una palabra. No
# importa: el token `<|inicio|>` que separa los cuentos le enseña al modelo dónde arranca uno nuevo.
#
# ## 6. Una ventana son muchos ejercicios
#
# En una ventana de 256 tokens no hay un ejercicio, hay **256**: predecir el 2.º a partir del
# 1.º, el 3.º a partir de los dos primeros... y el Transformer los resuelve **todos a la vez**
# en una sola pasada. Esa es una de las razones por las que desplazó a las redes recurrentes,
# que tenían que avanzar token a token (lo cuenta la introducción del artículo).

# %%
lote, pasos = proyecto.config["entreno"]["lote"], proyecto.config["entreno"]["pasos"]
print(f"Cada paso de entrenamiento: {lote} ventanas x {contexto} tokens = {lote * contexto:,} ejercicios")
print(f"En los {pasos:,} pasos de la receta: {lote * contexto * pasos:,} ejercicios")
print(f"El corpus de entrenamiento tiene {manifiesto['partes']['entreno']['tokens']:,} tokens")

# %% [markdown]
# ## Resumen
#
# * El ejercicio es siempre el mismo: **predecir el token siguiente**. Las respuestas salen
#   del propio texto: `y` es `x` desplazado un lugar.
# * Se aparta una **validación** que el modelo no ve al aprender, para medir si generaliza.
# * Se entrena por **lotes** de ventanas tomadas al azar del corpus.
#
# ## Ejercicios
#
# 1. Cambia el porcentaje de validación a 20. ¿Qué ganas y qué pierdes?
# 2. ¿Por qué es mala idea elegir la validación como «los últimos 100 cuentos del archivo»
#    si el archivo estuviera ordenado por tema?
# 3. Si un cuento apareciera repetido y una copia cayera en entrenamiento y otra en
#    validación, ¿qué le pasaría a la medida de validación? (El proyecto quita los duplicados.)
# 4. Con `lote=8` y `contexto=256`, ¿cuántos pasos hacen falta para ver tantos tokens como
#    tiene el corpus? ¿Los ve todos exactamente una vez? ¿Por qué no?
# 5. Abre `proyectos/curso/datos/manifiesto.json` y localiza cada número de esta clase.
#
# **Siguiente:** [Clase 3 · Embeddings y posición](03_embeddings_y_posicion.ipynb)
