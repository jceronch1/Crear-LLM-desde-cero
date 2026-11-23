# %% [markdown]
# # Clase 5 · El bloque Transformer: la pieza que se repite
#
# La atención mueve información **entre** tokens. Falta algo que la procese **dentro** de cada
# token, y falta una forma de apilar muchas capas sin que el entrenamiento se rompa. Las tres
# cosas juntas forman un **bloque**, y un Transformer es ese bloque repetido N veces.
#
# | | |
# |---|---|
# | **Duración** | 45 minutos |
# | **En el artículo** | §3.1 (atajos y normalización), §3.3 (red de avance), §5.4 (dropout) |
# | **En el código** | `cero/modelo.py`: `NormaRMS`, `RedDeAvance`, `Bloque` |
# | **Al terminar sabrás** | Para qué sirven los atajos residuales, la normalización y la red de avance |

# %% [markdown]
# ## 1. Atajos residuales
#
# En vez de sustituir el vector por el resultado de una capa, se le **suma**:
#
# ```
# sin atajo:   x = capa(x)
# con atajo:   x = x + capa(x)
# ```
#
# Así cada capa solo tiene que aprender una *corrección* sobre lo que ya había, y la señal
# (y el gradiente, al entrenar) tiene un camino directo de punta a punta. Veamos qué le pasa
# a una señal que atraviesa 30 capas con y sin atajo:

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
capas = [nn.Linear(64, 64) for _ in range(30)]
entrada = torch.randn(8, 64, requires_grad=True)


def atravesar(con_atajo):
    x = entrada
    for capa in capas:
        x = x + 0.3 * torch.tanh(capa(x)) if con_atajo else torch.tanh(capa(x))
    return x


for con_atajo in (False, True):
    salida = atravesar(con_atajo)
    gradiente, = torch.autograd.grad(salida.sum(), entrada)
    print(f"{'con' if con_atajo else 'sin'} atajo · gradiente que llega a la entrada tras 30 capas: {gradiente.abs().mean().item():.2e}")

# %% [markdown]
# Sin atajo, el gradiente llega al principio millones de veces más pequeño (fíjate en el
# exponente): las primeras capas no aprenden. Con atajo llega entero. Por eso el artículo pone «una conexión residual
# alrededor de cada una de las subcapas» (§3.1).
#
# ## 2. Normalización
#
# Al sumar capa tras capa, los números pueden crecer o encogerse. La normalización devuelve
# cada vector a un tamaño estándar.
#
# * **LayerNorm** (artículo): resta la media del vector y divide entre su desviación típica.
# * **RMSNorm** (modelos actuales): solo divide entre la raíz de la media de los cuadrados.
#   Más barata e igual de eficaz.
#
# $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{media}(x^2) + \varepsilon}} \cdot g$$
#
# (`g`, la «ganancia», es un número aprendido por dimensión; empieza en 1.)
#
# ### Tu turno · RMSNorm

# %%
from cero import taller


def mi_norma(x, eps=1e-6):
    # TU CÓDIGO: divide cada vector (última dimensión) entre la raíz de la media de sus cuadrados.
    # Pista: x.pow(2).mean(dim=-1, keepdim=True) es la media de los cuadrados de cada vector.
    raise NotImplementedError


taller.comprobar_norma_rms(mi_norma)

# %%
from cero.modelo import NormaRMS

x = torch.randn(3, 128) * torch.tensor([[0.1], [1.0], [50.0]])       # tres vectores de tamaños muy distintos
print("tamaño antes  :", [round(v, 2) for v in x.pow(2).mean(-1).sqrt().tolist()])
print("tamaño después:", [round(v, 2) for v in NormaRMS(128)(x).pow(2).mean(-1).sqrt().tolist()])

# %% [markdown]
# ### ¿Antes o después?
#
# | | Fórmula | Quién |
# |---|---|---|
# | **post-norma** | `x = Norma(x + Subcapa(x))` | el artículo de 2017 |
# | **pre-norma** | `x = x + Subcapa(Norma(x))` | GPT-2 y todos los posteriores |
#
# Con pre-norma el atajo queda completamente limpio (nada se interpone en el camino directo)
# y los modelos profundos entrenan con más estabilidad.
#
# ## 3. La red de avance
#
# La segunda mitad del bloque es una red pequeña de dos capas que se aplica **a cada token
# por separado** (§3.3). Ensancha el vector, aplica una función no lineal y lo vuelve a encoger:
#
# $$\text{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2$$
#
# `max(0, ·)` es la función **ReLU**. En el artículo el ancho interior es 4 veces el del modelo
# (512 → 2048 → 512). Aquí están la mayoría de los parámetros de un Transformer.
#
# Con los años ha cambiado la función no lineal:

# %%
valores = torch.tensor([-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0])
print("entrada      ", [f"{v:5.2f}" for v in valores.tolist()])
print("ReLU  (2017) ", [f"{v:5.2f}" for v in F.relu(valores).tolist()])
print("GELU  (GPT-2)", [f"{v:5.2f}" for v in F.gelu(valores).tolist()])
print("SiLU  (hoy)  ", [f"{v:5.2f}" for v in F.silu(valores).tolist()])

# %% [markdown]
# ReLU corta en seco todo lo negativo. GELU y SiLU hacen lo mismo con suavidad. Los modelos
# actuales usan **SwiGLU**, que añade una «puerta»: una segunda proyección decide, número a
# número, cuánto deja pasar de la primera.
#
# ```
# ReLU/GELU:  sale( activación( entra(x) ) )            ancho interior 4·d
# SwiGLU:     sale( silu(puerta(x)) * entra(x) )        ancho interior 8/3·d  (tres matrices en vez de dos)
# ```

# %%
from cero.config import ConfigModelo
from cero.modelo import RedDeAvance

for mlp in ("relu", "gelu", "swiglu"):
    cfg = ConfigModelo(d_modelo=128, mlp=mlp)
    red = RedDeAvance(cfg)
    print(f"{mlp:<7} ancho interior {cfg.d_oculta:>4} · {sum(p.numel() for p in red.parameters()):>7,} parámetros")

# %% [markdown]
# Las tres gastan casi lo mismo: SwiGLU usa un ancho menor justo para compensar su tercera matriz.
#
# ## 4. El bloque entero
#
# Así queda en el código del proyecto (versión pre-norma, la que usamos):
#
# ```python
# x = x + atencion(norma_1(x))     # mirar a los demás tokens
# x = x + red(norma_2(x))          # procesar lo que se ha traído
# ```

# %%
from cero.modelo import Bloque, tabla_rope

cfg = ConfigModelo(vocab=1024, contexto=64, d_modelo=128, n_capas=1, n_cabezas=4)
bloque = Bloque(cfg, capa=0)
x = torch.randn(2, 10, 128)
y = bloque(x, rope=tabla_rope(10, cfg.d_cabeza))
print("entra:", tuple(x.shape), "· sale:", tuple(y.shape), " (misma forma: por eso se pueden apilar)\n")
for nombre, pieza in bloque.named_children():
    print(f"  {nombre:<9} {sum(p.numel() for p in pieza.parameters()):>8,} parámetros")
print(f"  {'TOTAL':<9} {sum(p.numel() for p in bloque.parameters()):>8,}")

# %% [markdown]
# Un bloque recibe una lista de vectores y devuelve otra **de la misma forma**, un poco más
# «informada». Como entra lo mismo que sale, se pueden poner tantos como se quiera.
#
# ## 5. Dropout
#
# El artículo apaga al azar un 10 % de los números de cada subcapa durante el entrenamiento
# (§5.4) para que el modelo no dependa de ninguno en concreto. Ayuda cuando hay pocos datos
# para el tamaño del modelo. Los LLM actuales, que tienen datos de sobra, suelen ponerlo a 0;
# nosotros también (es el ajuste `dropout` de `ConfigModelo`).
#
# ## Resumen
#
# * Un bloque = **atención** + **red de avance**, cada una con su **atajo** y su **norma**.
# * Los atajos mantienen vivo el gradiente; la norma mantiene los números en su sitio.
# * La red de avance trabaja token a token y contiene la mayoría de los parámetros.
# * Lo que entra y lo que sale tienen la misma forma: el bloque se apila.
#
# ## Ejercicios
#
# 1. Calcula a mano los parámetros de un bloque con `d_modelo = 128`: 4 matrices de atención
#    de 128 × 128, 3 matrices de la red (el ancho interior es 344) y 2 normas de 128. ¿Coincide?
# 2. ¿Cuántos parámetros tendría el bloque con `d_modelo = 256`? ¿Se duplican o se cuadruplican?
# 3. En la sección 1, cambia 30 capas por 5 y por 100. ¿Cómo cambia el gradiente sin atajo?
# 4. Con LayerNorm, ¿por qué se normaliza cada token por separado y no la frase entera?
#    (Pista: piensa en la máscara causal.)
# 5. Crea el bloque con `pre_norma=False, norma="capa", mlp="relu", sesgos=True` (el de 2017)
#    y compara su número de parámetros con el moderno.
#
# **Siguiente:** [Clase 6 · El modelo completo](06_modelo.ipynb)
