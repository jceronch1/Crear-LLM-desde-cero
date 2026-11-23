# %% [markdown]
# # Clase 9 · Generar: cómo elige cada palabra
#
# El modelo no escribe. Solo da, para cada token del vocabulario, una puntuación de lo bien
# que encaja como siguiente. Escribir es **elegir uno, añadirlo y volver a preguntar**. Cómo
# se elige cambia mucho el resultado.
#
# | | |
# |---|---|
# | **Duración** | 45 minutos |
# | **En el artículo** | §6.1 (búsqueda en haz, la estrategia que usaba para traducir) |
# | **En el código** | `cero/generar.py` |
# | **Al terminar sabrás** | Qué hacen la temperatura, top-k y top-p, y por qué generar con caché es mucho más rápido |

# %%
import time

import torch

from cero import dispositivo, taller
from cero.generar import candidatos, completar, filtrar, generar
from cero.taller import RAPIDO

torch.set_grad_enabled(False)
DISPOSITIVO = dispositivo.elegir("auto")
proyecto = taller.asegurar(taller.proyecto_del_curso(), "base", DISPOSITIVO)
modelo, tok, _ = proyecto.cargar("base", DISPOSITIVO)

# %% [markdown]
# ## 1. Las opciones que baraja el modelo

# %%
texto = "Había una vez una gata curiosa que se llamaba Luna. Luna vivía en"
ids = [tok.id("<|inicio|>")] + tok.codificar_texto(texto)
for c in candidatos(modelo, tok, ids, cuantos=8):
    print(f"{c['texto']!r:<12} {c['prob']:>6.1%}  {'█' * round(40 * c['prob'])}")

# %% [markdown]
# ## 2. Elegir siempre el más probable
#
# Es lo más simple («voraz»), y da siempre el mismo texto:

# %%
print(texto + completar(modelo, tok, texto, max_nuevos=45, temperatura=0))

# %% [markdown]
# Correcto pero sin gracia: siempre el camino más trillado, y en modelos pequeños es fácil
# que entre en bucles. Para escribir texto variado se **sortea** según las probabilidades.
#
# ## 3. La temperatura
#
# Antes del softmax, los logits se dividen entre un número, la **temperatura**:
#
# * menor que 1 → las diferencias se agrandan: el favorito gana más (texto más previsible)
# * mayor que 1 → se aplanan: los improbables tienen más opciones (más sorpresa, más errores)
#
# ### Tu turno

# %%
def con_temperatura(logits, temperatura):
    # TU CÓDIGO: divide los logits entre la temperatura y devuelve el softmax (dim=-1).
    raise NotImplementedError


taller.comprobar_temperatura(con_temperatura)

# %%
logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
for t in (0.3, 1.0, 3.0):
    print(f"temperatura {t}: {[f'{p:.0%}' for p in (logits / t).softmax(-1).tolist()]}")

# %%
for t in (0.3, 0.8, 1.5):
    print(f"--- temperatura {t}")
    print(texto + completar(modelo, tok, texto, max_nuevos=40, temperatura=t, top_k=None, semilla=3), "\n")

# %% [markdown]
# ## 4. top-k y top-p
#
# Con temperatura alta, de vez en cuando sale un token absurdo y el texto descarrila. Dos
# formas de evitarlo, **recortando la cola** antes de sortear:
#
# * **top-k**: quedarse solo con los `k` tokens más probables.
# * **top-p** (muestreo de núcleo): quedarse con el grupo más pequeño de tokens que juntos
#   suman probabilidad `p`. Se adapta solo: pocos cuando el modelo está seguro, muchos cuando duda.
#
# ### Tu turno · top-k

# %%
def mi_top_k(logits, k):
    # TU CÓDIGO: deja a float("-inf") todos los logits menos los k mayores.
    # Pista: torch.topk(logits, k).values[-1] es el k-ésimo logit más alto (el umbral).
    raise NotImplementedError


taller.comprobar_top_k(mi_top_k)

# %%
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])
print("originales:", [f"{p:.0%}" for p in logits.softmax(-1).tolist()])
print("top_k = 2 :", [f"{p:.0%}" for p in filtrar(logits, top_k=2).softmax(-1).tolist()])
print("top_p = .8:", [f"{p:.0%}" for p in filtrar(logits, top_p=0.8).softmax(-1).tolist()])

# %% [markdown]
# Con el modelo: barra clara, lo que dice el modelo; barra oscura, lo que queda tras los mandos.

# %%
taller.ver_candidatos(candidatos(modelo, tok, ids, temperatura=0.7, top_k=5, cuantos=10), "temperatura 0,7 y top-k 5")

# %% [markdown]
# ## 5. El bucle completo
#
# Generar es esto. La función `generar` del proyecto hace lo mismo, con una mejora de velocidad.

# %%
generador = torch.Generator().manual_seed(1)
actuales = list(ids)
for _ in range(30):
    x = torch.tensor([actuales[-modelo.cfg.contexto:]], device=DISPOSITIVO)
    logits = modelo(x)[0][0, -1]                                    # 1. los logits de la ÚLTIMA posición
    probabilidades = filtrar(logits, temperatura=0.8, top_k=40).softmax(-1)
    siguiente = int(torch.multinomial(probabilidades.cpu(), 1, generator=generador))   # 2. sortear
    if siguiente in tok.especiales.values():
        break
    actuales.append(siguiente)                                      # 3. añadir y repetir
print(tok.decodificar(actuales[1:]))

# %% [markdown]
# ## 6. La caché: no repetir el trabajo
#
# Ese bucle tiene un derroche: para escribir el token 50 vuelve a pasar por el modelo los 49
# anteriores. Pero la atención es causal: las claves y los valores de los tokens ya escritos
# **no cambian nunca**. Si se guardan (la *caché KV*), cada token nuevo solo calcula lo suyo.

# %%
inicio_cuento = [tok.id("<|inicio|>")] + tok.codificar_texto("Había una vez")
cpu = modelo.to("cpu")                                # en CPU se aprecia mejor (abajo se explica por qué)
for cuantos in ((20, 40) if RAPIDO else (60, 240)):
    tiempos, textos = {}, {}
    for usar_cache in (False, True):
        t0 = time.perf_counter()
        textos[usar_cache] = list(generar(cpu, inicio_cuento, cuantos, semilla=5, usar_cache=usar_cache, parar=()))
        tiempos[usar_cache] = time.perf_counter() - t0
    print(f"{cuantos:>4} tokens · sin caché {cuantos / tiempos[False]:5.0f} tokens/s · con caché {cuantos / tiempos[True]:5.0f} tokens/s"
          f" · {tiempos[False] / tiempos[True]:.1f} veces más rápido · mismo texto: {textos[False] == textos[True]}")
modelo.to(DISPOSITIVO)

# %% [markdown]
# El texto es idéntico y la ventaja **crece con la longitud**: sin caché, cada token nuevo
# cuesta más que el anterior; con caché, todos cuestan lo mismo.
#
# Con un modelo tan pequeño en GPU la diferencia casi no se nota: la tarjeta tarda más en
# *recibir* cada orden que en hacer la cuenta. En modelos grandes y textos largos, la caché es
# imprescindible: todos los servicios de LLM la usan, y es lo que más memoria ocupa al servirlos.
#
# ## 7. La ventana de contexto
#
# El modelo solo ve `contexto` tokens (256 en la receta nano). Si el texto crece más, lo más
# antiguo deja de verse: el modelo literalmente **lo olvida**.
#
# ## 8. ¿Y la búsqueda en haz del artículo?
#
# Para traducir, el artículo no sorteaba: exploraba varias continuaciones a la vez y se
# quedaba con la frase completa más probable (*beam search*, §6.1). Tiene sentido cuando hay
# una respuesta «correcta», como en una traducción. Para escribir texto abierto produce
# resultados repetitivos, y por eso los LLM de hoy muestrean.
#
# ## Resumen
#
# * Generar = logits de la última posición → sorteo → añadir → repetir.
# * **Temperatura**: cuánto riesgo. **top-k / top-p**: cortar la cola de tokens improbables.
# * La **caché KV** evita recalcular lo ya escrito.
#
# ## Ejercicios
#
# 1. Genera el mismo comienzo con temperatura 0,1 y con 2,0. Describe la diferencia.
# 2. Con `top_k = 1`, ¿importa la temperatura? ¿Por qué?
# 3. Genera dos veces con la misma semilla y dos veces con semillas distintas. ¿Qué controla
#    la semilla: lo que el modelo sabe o el sorteo?
# 4. Dale un comienzo que no tenga nada que ver con cuentos («El motor del coche»). ¿Adónde
#    lleva el texto? ¿Qué te dice eso de lo que ha aprendido?
# 5. Abre `cero lab`, ve a «Generar» y usa «Añadir un token» varias veces mirando cómo
#    cambian los candidatos.
#
# **Siguiente:** [Clase 10 · De LLM a chatbot](10_chat.ipynb)
