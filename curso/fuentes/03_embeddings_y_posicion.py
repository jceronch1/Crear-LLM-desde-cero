# %% [markdown]
# # Clase 3 · Embeddings y posición: de un número a un vector con sentido
#
# El token 317 no «significa» nada: es una etiqueta. Para que una red pueda trabajar con él
# hay que convertirlo en un **vector** (una lista de números) que sí pueda sumar, comparar y
# mezclar. Y hay que decirle, de alguna forma, **en qué posición** está cada token.
#
# | | |
# |---|---|
# | **Duración** | 45 minutos |
# | **En el artículo** | §3.4 (embeddings) y §3.5 (codificación posicional) |
# | **En el código** | `cero/modelo.py`: `tabla_sinusoidal`, `tabla_rope`, `rotar` |
# | **Al terminar sabrás** | Qué es una tabla de embeddings y tres formas de indicar el orden: sinusoidal, aprendida y RoPE |

# %% [markdown]
# ## 1. La tabla de embeddings
#
# Es una matriz con **una fila por cada token del vocabulario**. Cada fila tiene `d_modelo`
# números. Convertir un token en vector es, simplemente, ir a su fila.

# %%
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_grad_enabled(False)                   # hoy solo miramos: no hace falta calcular gradientes
vocab, d_modelo = 1024, 8                       # 8 para que quepa en pantalla; el proyecto usa 128
tabla = nn.Embedding(vocab, d_modelo)
print("Forma de la tabla:", tuple(tabla.weight.shape), f"= {tabla.weight.numel():,} parámetros")

tokens = torch.tensor([317, 5, 317])
vectores = tabla(tokens)
print("\nTres tokens -> tres vectores:", tuple(vectores.shape))
print(vectores)

# %% [markdown]
# El token 317 aparece dos veces y recibe **el mismo vector** las dos veces. Ahora mismo son
# números al azar: no significan nada. Al entrenar, cada fila se va moviendo hasta que los
# tokens que se usan de forma parecida acaban teniendo vectores parecidos (lo veremos con el
# modelo entrenado en la clase 8).
#
# > **En el artículo (§3.4).** «Usamos embeddings aprendidos para convertir los tokens de
# > entrada en vectores de dimensión d_model.» En su modelo base, d_model = 512.

# %% [markdown]
# ## 2. El problema del orden
#
# Como veremos en la clase 4, la atención trata el texto como una **bolsa de vectores**: no
# sabe cuál va delante de cuál. Mira qué pasa si solo sumamos los vectores de una frase:

# %%
frase_a = torch.tensor([10, 20, 30])            # digamos: «perro», «muerde», «niño»
frase_b = torch.tensor([30, 20, 10])            # «niño», «muerde», «perro»
print("Suma de la frase A:", tabla(frase_a).sum(0)[:4])
print("Suma de la frase B:", tabla(frase_b).sum(0)[:4])

# %% [markdown]
# Idénticas. Hay que añadir al vector de cada token algo que dependa de **su posición**.
# Vamos a ver las tres soluciones que se han usado, en orden histórico.
#
# ## 3. Senos y cosenos: la del artículo (2017)
#
# El artículo fabrica, para cada posición, un vector fijo con esta fórmula (§3.5):
#
# $$PE_{(pos,\,2i)} = \operatorname{sen}\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad
#   PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
#
# `pos` es la posición del token e `i` recorre las parejas de columnas. Ese vector se **suma**
# al embedding del token.
#
# ### Tu turno · la tabla sinusoidal
# Completa la función para que devuelva una tabla de forma `(largo, d)` con esa fórmula.

# %%
from cero import taller


def mi_tabla(largo, d):
    posicion = torch.arange(largo).float().unsqueeze(1)               # columna: 0, 1, 2... (largo, 1)
    divisor = 10000 ** (torch.arange(0, d, 2).float() / d)            # un valor por cada pareja de columnas
    tabla = torch.zeros(largo, d)
    # TU CÓDIGO: las columnas pares (0, 2, 4...) llevan el seno de posicion / divisor
    # y las impares (1, 3, 5...) el coseno.  Pista: tabla[:, 0::2] son las columnas pares.
    raise NotImplementedError
    return tabla


taller.comprobar_sinusoidal(mi_tabla)

# %%
from cero.modelo import tabla_sinusoidal

posiciones = tabla_sinusoidal(64, 64)
taller.ver_tabla(posiciones, "Codificación sinusoidal: cada fila es el vector de una posición")

# %% [markdown]
# Las primeras columnas oscilan muy deprisa (cambian de una posición a la siguiente) y las
# últimas muy despacio. Es como las agujas de un reloj: el segundero distingue instantes
# cercanos y la de las horas, los lejanos. Juntas dan a cada posición una «hora» única.
#
# Una consecuencia útil: las posiciones cercanas tienen vectores parecidos.

# %%
parecido = posiciones @ posiciones[20]           # producto escalar de la posición 20 con todas
for pos in (0, 10, 18, 19, 20, 21, 22, 30, 60):
    print(f"posición 20 frente a {pos:>2}: {parecido[pos].item():6.1f}  {'█' * int(parecido[pos].item())}")

# %% [markdown]
# ## 4. Posiciones aprendidas: la de GPT-2 (2019)
#
# Otra opción: una segunda tabla de embeddings, con una fila por **posición**, que se aprende
# como todo lo demás. El artículo también la probó y obtuvo «resultados casi idénticos»
# (§3.5, fila E de su Tabla 3).

# %%
contexto = 16
tabla_posiciones = nn.Embedding(contexto, d_modelo)
x = tabla(tokens) + tabla_posiciones(torch.arange(len(tokens)))
print("Ahora el token 317 de la posición 0 y el de la posición 2 ya no son iguales:")
print(x[0][:4], "\n", x[2][:4], sep="")

# %% [markdown]
# Su límite: solo existen las posiciones para las que hay fila. Un modelo entrenado con
# `contexto = 16` no tiene vector para la posición 17.
#
# ## 5. RoPE: la de hoy
#
# Las dos anteriores dicen «estás en la posición 7». Pero al lenguaje le suele importar más
# la **distancia**: «el adjetivo va justo detrás del nombre», no «el adjetivo está en la 7».
#
# RoPE (*Rotary Position Embedding*, 2021) no suma nada al embedding. Actúa **dentro de la
# atención**: gira los vectores un ángulo proporcional a su posición. Empecemos con dos
# dimensiones, que se puede dibujar: girar el punto (1, 0).

# %%
import math


def girar(vector, angulo):
    x1, x2 = vector
    return (x1 * math.cos(angulo) - x2 * math.sin(angulo), x1 * math.sin(angulo) + x2 * math.cos(angulo))


for posicion in range(5):
    gx, gy = girar((1.0, 0.0), posicion * 0.5)                # medio radián por posición
    print(f"posición {posicion}: ({gx:5.2f}, {gy:5.2f})   ángulo {posicion * 0.5:.1f} rad")

# %% [markdown]
# La propiedad que lo hace útil: si giro un vector `q` según su posición `m` y otro vector `k`
# según su posición `n`, **su producto escalar solo depende de la distancia `m − n`**, no de
# dónde estén. (Girar los dos lo mismo no cambia el ángulo entre ellos.)
#
# En el modelo, el vector de cada cabeza se parte en parejas de números y cada pareja gira a
# una velocidad distinta: las mismas frecuencias de la tabla sinusoidal de 2017. Comprobémoslo
# con las funciones del proyecto:

# %%
from cero.modelo import rotar, tabla_rope

torch.manual_seed(0)
cos, sin = tabla_rope(largo=64, d_cabeza=16)
q, k = torch.randn(16), torch.randn(16)


def producto(m, n):
    return float((rotar(q, cos[m], sin[m]) * rotar(k, cos[n], sin[n])).sum())


print("q en la posición  3, k en la  1 (distancia 2):", round(producto(3, 1), 4))
print("q en la posición 13, k en la 11 (distancia 2):", round(producto(13, 11), 4))
print("q en la posición 60, k en la 58 (distancia 2):", round(producto(60, 58), 4))
print("q en la posición 13, k en la 10 (distancia 3):", round(producto(13, 10), 4))

# %% [markdown]
# Misma distancia, mismo resultado, estén donde estén. La atención «ve» posiciones relativas,
# y **no hay ningún parámetro de posición que aprender**.
#
# ## 6. En el proyecto
#
# Las tres opciones están en `cero/modelo.py` y se eligen con un interruptor:

# %%
from cero.config import ARQUITECTURAS, ConfigModelo
from cero.modelo import GPT

for nombre, interruptores in ARQUITECTURAS.items():
    cfg = ConfigModelo(vocab=1024, contexto=256, d_modelo=128, n_capas=4, n_cabezas=4, **interruptores)
    modelo = GPT(cfg)
    print(f"{nombre:<10} posición = {cfg.posicion:<11} parámetros de posición: {modelo.reparto()['posiciones']:>6,}")

# %% [markdown]
# Solo la versión «aprendida» gasta parámetros en la posición (256 × 128). La sinusoidal es
# una tabla fija y RoPE son rotaciones calculadas.
#
# ## Resumen
#
# * La **tabla de embeddings** convierte cada token en un vector que se aprende.
# * Sin información de posición, el modelo no distingue el orden de las palabras.
# * Tres soluciones: sumar una tabla fija de **senos y cosenos** (2017), sumar una tabla
#   **aprendida** (GPT-2) o **girar** los vectores dentro de la atención (RoPE, hoy).
#
# ## Ejercicios
#
# 1. La tabla de embeddings de nuestro modelo tiene 1.024 filas de 128 números. ¿Cuántos
#    parámetros son? ¿Y la de GPT-2, con 50.257 tokens y 768 dimensiones?
# 2. En la sección 3, cambia el 10000 de la fórmula por 10 y vuelve a dibujar la tabla. ¿Qué
#    le pasa a las últimas columnas? ¿Por qué dejaría de servir para textos largos?
# 3. Con posiciones aprendidas y `contexto = 256`, ¿qué ocurre si le pasas al modelo un texto
#    de 300 tokens? ¿Y con RoPE? (Pista: mira `GPT.estados` en `cero/modelo.py`.)
# 4. En la sección 5, comprueba que `producto(5, 5)`, `producto(40, 40)` y el producto sin
#    girar (`float((q * k).sum())`) coinciden. ¿Por qué?
# 5. ¿Por qué RoPE gira las consultas y las claves, pero no los valores? (Vuelve a esta
#    pregunta después de la clase 4.)
#
# **Siguiente:** [Clase 4 · La atención](04_atencion.ipynb)
