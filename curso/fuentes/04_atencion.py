# %% [markdown]
# # Clase 4 · La atención: cómo usa cada token a los anteriores
#
# Es la pieza que da nombre al artículo. Para predecir lo que viene después de «Luna vivía en
# el bosque. A Luna le gustaba», al modelo le conviene **mirar atrás** y fijarse en «Luna».
# La atención es el mecanismo que decide, para cada token, cuánto mirar a cada uno de los anteriores.
#
# | | |
# |---|---|
# | **Duración** | 60 minutos |
# | **En el artículo** | §3.2: ecuación (1), Figura 2, varias cabezas (§3.2.2) y Tabla 1 |
# | **En el código** | `cero/modelo.py`: `atencion`, `mascara_causal`, `Atencion` |
# | **Al terminar sabrás** | Calcular la atención a mano, por qué se enmascara el futuro y qué son las cabezas |

# %% [markdown]
# ## 1. La idea: consultas, claves y valores
#
# De cada token se sacan tres vectores:
#
# | Vector | Nombre | Papel |
# |---|---|---|
# | **Q** | consulta (*query*) | «qué estoy buscando» |
# | **K** | clave (*key*) | «qué ofrezco a quien busque» |
# | **V** | valor (*value*) | «qué información entrego si me eligen» |
#
# Se compara la consulta de un token con las claves de todos (producto escalar). Las
# puntuaciones se convierten en porcentajes y con ellos se hace una **media ponderada de los
# valores**. La ecuación (1) del artículo lo dice en una línea:
#
# $$\text{Atención}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$
#
# Vamos a calcularla con números, paso a paso.

# %%
import math

import torch
import torch.nn.functional as F

torch.manual_seed(3)
palabras = ["Luna", "vivía", "en", "el", "bosque"]
T, d = len(palabras), 4                              # 5 tokens, vectores de 4 números


def mostrar(titulo, matriz, columnas=None):
    print(titulo)
    columnas = columnas or [""] * matriz.shape[1]
    print(" " * 9 + "".join(f"{c:>8}" for c in columnas))
    for nombre, fila in zip(palabras, matriz):
        print(f"{nombre:>8} " + "".join("    -inf" if v == float("-inf") else f"{v:8.2f}" for v in fila.tolist()))
    print()


x = torch.randn(T, d)                                # los vectores que llegan (embeddings)
W_q, W_k, W_v = (torch.randn(d, d) * 0.7 for _ in range(3))    # tres matrices que SE APRENDEN
Q, K, V = x @ W_q, x @ W_k, x @ W_v
mostrar("Consultas Q = x · W_q", Q)

# %% [markdown]
# ### Paso 1 · Puntuaciones: cada consulta contra cada clave

# %%
puntos = Q @ K.T / math.sqrt(d)
mostrar("Puntuaciones  (fila = quién pregunta, columna = a quién mira)", puntos, palabras)

# %% [markdown]
# ### Paso 2 · Prohibido mirar al futuro
#
# Entrenamos al modelo a predecir el token siguiente. Si al calcular la posición 2 pudiera
# mirar la posición 3, **vería la respuesta**: aprendería a copiarla y no serviría para
# escribir, porque al escribir el futuro todavía no existe.
#
# La **máscara causal** lo impide: cada token solo puede mirar a sí mismo y a los anteriores.
#
# #### Tu turno · la máscara
# Devuelve una matriz `T × T` de `True`/`False`: `True` donde SÍ se permite mirar.

# %%
from cero import taller


def mi_mascara(T):
    # TU CÓDIGO: la fila i (quien pregunta) puede mirar las columnas 0, 1, ..., i.
    # Pista: torch.ones(T, T).tril() deja unos en el triángulo inferior; .bool() lo convierte.
    raise NotImplementedError


taller.comprobar_mascara(mi_mascara)

# %%
from cero.modelo import mascara_causal

mascara = mascara_causal(T, T)
puntos_tapados = puntos.masked_fill(~mascara, float("-inf"))
mostrar("Puntuaciones con el futuro tapado", puntos_tapados, palabras)

# %% [markdown]
# ### Paso 3 · De puntuaciones a porcentajes (softmax)
#
# El softmax convierte una lista de números en porcentajes que suman 1. Un −infinito se
# convierte en un 0 exacto: por eso se tapa con −infinito.

# %%
pesos = F.softmax(puntos_tapados, dim=-1)
mostrar("Pesos de atención (cada fila suma 1)", pesos, palabras)

# %% [markdown]
# ### Paso 4 · Mezclar los valores

# %%
salida = pesos @ V
mostrar("Salida = pesos · V", salida)

# %% [markdown]
# Cada fila de la salida es una mezcla de los valores de los tokens que esa posición podía
# mirar. El vector de «bosque» ya no habla solo de «bosque»: lleva dentro información de
# «Luna», «vivía»... en la proporción que marcan sus pesos. Eso es **contexto**.
#
# ### Tu turno · la atención completa
# Escribe los cuatro pasos en una función. `q`, `k`, `v` pueden traer dimensiones de más
# delante (lote, cabezas): usa `k.transpose(-2, -1)` para trasponer y `dim=-1` en el softmax.

# %%
def mi_atencion(q, k, v, mascara):
    # TU CÓDIGO:
    # 1. puntos = q · kᵀ / raíz(d_k)        (d_k = q.size(-1))
    # 2. tapar con -inf donde la máscara es False     (puntos.masked_fill(~mascara, float("-inf")))
    # 3. pesos = softmax(puntos)
    # 4. devolver pesos · v
    raise NotImplementedError


taller.comprobar_atencion(mi_atencion)

# %% [markdown]
# PyTorch trae esta misma cuenta en una función optimizada. Nuestra versión a mano y la suya
# deben coincidir:

# %%
from cero.modelo import atencion

a_mano, _ = atencion(Q, K, V, mascara)
de_pytorch = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print("Diferencia máxima entre la versión a mano y la de PyTorch:", float((a_mano - de_pytorch).abs().max()))

# %% [markdown]
# ## 2. ¿Por qué se divide entre √d_k?
#
# El artículo lo explica en una nota al pie: si los vectores tienen `d_k` números, sus
# productos escalares crecen como √d_k. Con números grandes el softmax **se satura**: da casi
# todo el peso a un solo token, y entonces el gradiente casi desaparece y no se aprende.

# %%
print(f"{'d_k':>6} {'tamaño típico de q·k':>22} {'peso máximo SIN dividir':>26} {'peso máximo dividiendo':>24}")
for d_k in (4, 16, 64, 256, 1024):
    q, k = torch.randn(2000, d_k), torch.randn(2000, 8, d_k)
    productos = torch.einsum("nd,nkd->nk", q, k)              # 2000 consultas contra 8 claves cada una
    sin = productos.softmax(-1).max(-1).values.mean()
    con = (productos / math.sqrt(d_k)).softmax(-1).max(-1).values.mean()
    print(f"{d_k:>6} {productos.std().item():>22.1f} {sin.item():>25.0%} {con.item():>23.0%}")

# %% [markdown]
# Sin dividir, con vectores de 256 números casi el 100 % del peso se va a un único token.
# Dividiendo, el reparto es el mismo sea cual sea el tamaño.
#
# ## 3. Varias cabezas
#
# Una sola atención da **un** reparto de pesos por token. Pero conviene mirar varias cosas a
# la vez: el nombre del protagonista, la palabra anterior, si hay una raya de diálogo abierta...
#
# La solución del artículo (§3.2.2): partir los `d_modelo` números en `h` grupos y hacer `h`
# atenciones en paralelo, cada una con sus propias matrices. Se llaman **cabezas**.
#
# > «La atención con varias cabezas permite al modelo atender conjuntamente a información de
# > distintos subespacios de representación en distintas posiciones.»
#
# El truco está en reordenar el tensor: `(lote, tokens, d_modelo)` → `(lote, cabezas, tokens, d_cabeza)`.

# %%
lote, tokens, d_modelo, cabezas = 2, 5, 16, 4
d_cabeza = d_modelo // cabezas
x = torch.randn(lote, tokens, d_modelo)
por_cabezas = x.view(lote, tokens, cabezas, d_cabeza).transpose(1, 2)
print("antes  :", tuple(x.shape), "            (lote, tokens, d_modelo)")
print("después:", tuple(por_cabezas.shape), "         (lote, cabezas, tokens, d_cabeza)")

# %% [markdown]
# A partir de ahí la función de atención es la misma: las dimensiones de delante (lote y
# cabezas) simplemente «viajan». Al final se vuelven a juntar las cabezas y se mezclan con
# una última matriz, W_O. Es el módulo `Atencion` del proyecto:

# %%
from cero.config import ConfigModelo
from cero.modelo import Atencion, tabla_rope

cfg = ConfigModelo(vocab=1024, contexto=64, d_modelo=128, n_capas=1, n_cabezas=4)
modulo = Atencion(cfg, capa=0)
x = torch.randn(1, 12, 128)
cos, sin = tabla_rope(12, cfg.d_cabeza)
ver = []                                              # aquí dejará los pesos para poder mirarlos
y = modulo(x, rope=(cos, sin), ver=ver)
print("entra:", tuple(x.shape), "· sale:", tuple(y.shape), "· pesos:", tuple(ver[0].shape))
for nombre, matriz in modulo.named_parameters():
    print(f"   {nombre:<12} {tuple(matriz.shape)}")
print("Parámetros de la atención:", sum(p.numel() for p in modulo.parameters()), "= 4 · 128²")

# %% [markdown]
# Cuatro matrices: las W^Q, W^K, W^V y W^O del artículo. Con 4 cabezas de 32 dimensiones se
# gastan los **mismos parámetros** que con una cabeza de 128: se reparte el mismo presupuesto
# en cuatro miradas distintas.
#
# Así se ven los pesos de un modelo **sin entrenar** (ruido repartido casi por igual). En la
# clase 8 miraremos los del modelo entrenado, y ahí sí se ve qué busca cada cabeza.

# %%
taller.ver_atencion(ver[0][0, 0], [str(i) for i in range(12)], "Cabeza 1 de un modelo sin entrenar")

# %% [markdown]
# ## 4. Lo que cuesta
#
# Cada token mira a todos los anteriores: con `T` tokens hay `T × T` puntuaciones. El coste
# crece con el **cuadrado** de la longitud (Tabla 1 del artículo: O(n²·d) por capa). A cambio,
# cualquier token está a **un solo paso** de cualquier otro, por lejos que esté. En una red
# recurrente la información tenía que pasar de token en token.

# %%
import time

q = torch.randn(1, 4, 2048, 32)
for T in (256, 512, 1024, 2048):
    inicio = time.perf_counter()
    for _ in range(5):
        F.scaled_dot_product_attention(q[:, :, :T], q[:, :, :T], q[:, :, :T], is_causal=True)
    print(f"{T:>5} tokens: {T * T:>9,} puntuaciones por cabeza · {(time.perf_counter() - inicio) / 5 * 1000:6.1f} ms")

# %% [markdown]
# Por eso todos los modelos tienen una **ventana de contexto** limitada: doblar la ventana
# cuesta el cuádruple en esta parte.
#
# ## Resumen
#
# * Atención = comparar consultas con claves, tapar el futuro, softmax y promediar valores.
# * La **máscara causal** es lo que permite entrenar a «predecir el siguiente» sin trampas.
# * Se divide entre √d_k para que el softmax no se sature.
# * **Varias cabezas** = varias atenciones en paralelo con el mismo presupuesto de parámetros.
# * El coste crece con el cuadrado de la longitud del texto.
#
# ## Ejercicios
#
# 1. En el paso 3, multiplica las puntuaciones por 10 antes del softmax. ¿Qué les pasa a los
#    pesos? Relaciónalo con la sección 2.
# 2. Quita la máscara (usa `puntos` en vez de `puntos_tapados`). ¿Qué fila cambia más? ¿Cuál
#    no cambia nada, y por qué?
# 3. La primera fila de los pesos es siempre `1, 0, 0, ...`. ¿Por qué?
# 4. Con `d_modelo = 128`, ¿cuántos parámetros tiene la atención con 1, 4 y 8 cabezas?
#    Compruébalo creando el módulo con cada valor.
# 5. Vuelve al ejercicio 5 de la clase 3: ¿por qué RoPE gira Q y K pero no V?
#
# **Siguiente:** [Clase 5 · El bloque Transformer](05_bloque.ipynb)
