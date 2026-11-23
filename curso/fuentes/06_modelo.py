# %% [markdown]
# # Clase 6 · El modelo completo
#
# Ya tenemos todas las piezas. Hoy las montamos, seguimos a un lote de punta a punta y
# contamos dónde están los parámetros.
#
# | | |
# |---|---|
# | **Duración** | 40 minutos |
# | **En el artículo** | Figura 1, §3.4 (pesos compartidos) y Tabla 3 (tamaños) |
# | **En el código** | `cero/modelo.py`: `GPT` · `cero/config.py`: `ConfigModelo` |
# | **Al terminar sabrás** | Cómo encajan las piezas, qué son los logits y por qué un modelo sin entrenar tiene pérdida ln(V) |

# %% [markdown]
# ## 1. El montaje
#
# ```
# tokens ─► tabla de embeddings ─► N × Bloque ─► norma ─► capa de salida ─► logits
# ```
#
# Toda la forma del modelo cabe en una ficha de configuración:

# %%
import math

import torch

from cero.config import ConfigModelo
from cero.modelo import GPT

torch.manual_seed(0)
torch.set_grad_enabled(False)                    # hoy solo miramos: no hace falta calcular gradientes
cfg = ConfigModelo(vocab=1024, contexto=256, d_modelo=128, n_capas=4, n_cabezas=4)
modelo = GPT(cfg)
print(cfg, "\n")
print(f"{modelo.n_parametros():,} parámetros")

# %%
reparto = modelo.reparto()
for pieza, cuantos in reparto.items():
    if pieza != "total" and cuantos:
        print(f"{pieza:<14} {cuantos:>9,}  {100 * cuantos / reparto['total']:5.1f} %  {'█' * round(40 * cuantos / reparto['total'])}")

# %% [markdown]
# Más de la mitad está en las redes de avance, algo más de una cuarta parte en la atención y
# el resto en la tabla de embeddings.
#
# ## 2. Un lote, de punta a punta

# %%
x = torch.randint(0, cfg.vocab, (2, 10))                 # 2 textos de 10 tokens
vectores = modelo.embeddings(x)
estados = modelo.estados(x)                              # embeddings + los 4 bloques + norma final
logits = modelo.salida(estados)
print("tokens      ", tuple(x.shape), "         números enteros")
print("embeddings  ", tuple(vectores.shape), "    un vector por token")
print("tras bloques", tuple(estados.shape), "    los mismos vectores, ya con contexto")
print("logits      ", tuple(logits.shape), "   una puntuación por cada token del vocabulario")

# %% [markdown]
# Los **logits** son la salida del modelo: para cada posición, 1.024 puntuaciones, una por
# cada token que podría venir después. El softmax las convierte en probabilidades.

# %%
probabilidades = logits[0, -1].softmax(-1)               # tras el último token del primer texto
print("Suman:", round(float(probabilidades.sum()), 4))
print("La más alta:", round(float(probabilidades.max()), 5), "· la más baja:", round(float(probabilidades.min()), 5))
print("Si todas fueran iguales valdrían 1/1024 =", round(1 / 1024, 5))

# %% [markdown]
# Sin entrenar, todas las probabilidades son parecidas: el modelo no tiene ni idea.
#
# ## 3. Pesos atados
#
# La tabla de embeddings convierte token → vector. La capa de salida hace el camino inverso:
# vector → puntuación de cada token. El artículo (§3.4) usa **la misma matriz** para las dos:
#
# > «Compartimos la misma matriz de pesos entre las dos capas de embeddings y la
# > transformación lineal previa al softmax.»

# %%
print("¿Son el mismo objeto?", modelo.salida.weight is modelo.embeddings.weight)
suelto = GPT(ConfigModelo(vocab=1024, contexto=256, d_modelo=128, n_capas=4, n_cabezas=4, atar_pesos=False))
print(f"Con pesos atados: {modelo.n_parametros():,} · sin atar: {suelto.n_parametros():,} "
      f"(+{suelto.n_parametros() - modelo.n_parametros():,} = 1024 × 128)")

# %% [markdown]
# En un modelo pequeño el ahorro es enorme. Con el vocabulario de GPT-2 (50.257 tokens) y
# este ancho, la tabla sola tendría 6,4 millones de parámetros: siete veces todo nuestro modelo.
# Por eso también importa que el tokenizador esté hecho a la medida del corpus (clase 1).
#
# ## 4. La pérdida de un modelo que no sabe nada
#
# La **pérdida** mide lo mal que predice: es `−ln(probabilidad que le dio al token correcto)`,
# promediada sobre todas las posiciones (se llama *entropía cruzada*).
#
# Si el modelo reparte la probabilidad por igual entre los V tokens, a cada uno le da 1/V, y
# la pérdida es `−ln(1/V) = ln(V)`.

# %%
y = torch.randint(0, cfg.vocab, (2, 10))
_, perdida = modelo(x, y)
print(f"Pérdida del modelo sin entrenar: {perdida.item():.3f}")
print(f"ln(1024) = {math.log(1024):.3f}")

# %% [markdown]
# Ese número es el punto de partida de todo entrenamiento. **Todo lo que baje de ahí es algo
# que el modelo ha aprendido.**
#
# Y esto es lo que escribe ahora mismo:

# %%
from cero import taller
from cero.generar import completar

proyecto = taller.asegurar(taller.proyecto_del_curso(), "datos")
tok = proyecto.tokenizador()
sin_entrenar = GPT(proyecto.config_modelo())
print(repr(completar(sin_entrenar, tok, "Había una vez", max_nuevos=30, semilla=1, sin_parar=True)))

# %% [markdown]
# Trozos al azar. La arquitectura está completa, pero sus 900.000 números son ruido. En la
# clase 7 los ajustaremos.
#
# ## 5. Tres generaciones con el mismo código
#
# Los interruptores de `ConfigModelo` permiten montar el decodificador del artículo de 2017,
# un GPT-2 o un modelo actual:

# %%
from cero.config import ARQUITECTURAS

print(f"{'arquitectura':<11} {'posición':<11} {'norma':<6} {'antes':<6} {'red':<7} {'parámetros':>11}")
for nombre, interruptores in ARQUITECTURAS.items():
    c = ConfigModelo(vocab=1024, contexto=256, d_modelo=128, n_capas=4, n_cabezas=4, **interruptores)
    print(f"{nombre:<11} {c.posicion:<11} {c.norma:<6} {'sí' if c.pre_norma else 'no':<6} {c.mlp:<7} {GPT(c).n_parametros():>11,}")

# %% [markdown]
# En la clase 12 las entrenaremos las tres con los mismos datos para compararlas.
#
# ## 6. Cuestión de escala
#
# | Modelo | Capas | Ancho | Cabezas | Parámetros |
# |---|---:|---:|---:|---:|
# | Nuestra receta `nano` | 4 | 128 | 4 | 0,9 millones |
# | Nuestra receta `mini` | 8 | 384 | 6 | 16 millones |
# | Transformer «base» del artículo (Tabla 3) | 6 + 6 | 512 | 8 | 65 millones |
# | Transformer «big» del artículo (Tabla 3) | 6 + 6 | 1024 | 16 | 213 millones |
# | GPT-2 pequeño (2019) | 12 | 768 | 12 | 124 millones |
#
# Los modelos que usas a diario tienen miles de veces más. **La arquitectura es la misma**
# que acabas de montar: cambian el tamaño, los datos y el tiempo de entrenamiento.
#
# ## Resumen
#
# * El modelo completo: embeddings → N bloques → norma → capa de salida → **logits**.
# * La matriz de embeddings y la de salida son la misma (pesos atados).
# * Un modelo sin entrenar reparte por igual: su pérdida es **ln(V)**.
#
# ## Ejercicios
#
# 1. Sin ejecutar nada: con `vocab = 4096`, ¿cuál es la pérdida de un modelo sin entrenar?
# 2. Crea un modelo con `n_capas=8`. ¿Cuántos parámetros añade cada bloque? ¿Y si en vez de
#    eso doblas `d_modelo`?
# 3. ¿Qué forma tienen los logits para un lote de 16 textos de 256 tokens? ¿Cuántos números son?
# 4. Cambia la semilla del ejemplo de la sección 4. ¿Cambia el texto? ¿Mejora? ¿Por qué?
# 5. Lee `GPT.forward` en `cero/modelo.py` (son 6 líneas) y localiza cada paso de esta clase.
#
# **Siguiente:** [Clase 7 · Entrenar](07_entrenar.ipynb)
