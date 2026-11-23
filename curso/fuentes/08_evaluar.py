# %% [markdown]
# # Clase 8 · Evaluar y mirar por dentro
#
# El modelo ya escribe. Ahora toca la pregunta incómoda: **¿ha aprendido, o se ha aprendido
# de memoria los cuentos?** Y otra más interesante: **¿qué ha aprendido exactamente?**
#
# | | |
# |---|---|
# | **Duración** | 50 minutos |
# | **En el artículo** | Tabla 3 (perplejidad), Figuras 3 a 5 (mapas de atención) |
# | **En el código** | `cero/entrenar.py`: `medir_perdida` · `cero/modelo.py`: `GPT.mirar` |
# | **Al terminar sabrás** | Qué miden la pérdida y la perplejidad, qué es el sobreajuste y cómo leer un mapa de atención |

# %%
import math

import torch

from cero import datos, dispositivo, taller
from cero.entrenar import medir_perdida
from cero.taller import RAPIDO

torch.set_grad_enabled(False)                       # hoy casi todo es mirar (entrenar los vuelve a activar solo)
DISPOSITIVO = dispositivo.elegir("auto")
proyecto = taller.asegurar(taller.proyecto_del_curso(), "base", DISPOSITIVO)
modelo, tok, ficha = proyecto.cargar("base", DISPOSITIVO)
print(f"Modelo base del proyecto «{proyecto.nombre}», guardado en el paso {ficha['paso']}")

# %% [markdown]
# ## 1. Pérdida, perplejidad y bits por carácter
#
# Se mide en la **validación**: cuentos que el modelo no vio al entrenar.

# %%
contexto = modelo.cfg.contexto
lotes = datos.Lotes(proyecto.datos / "validacion.bin", contexto, 16, semilla=4321, dispositivo=DISPOSITIVO)
perdida = medir_perdida(modelo, lotes, 4 if RAPIDO else 30, DISPOSITIVO)
caracteres_por_token = proyecto.estado()["datos"]["caracteres_por_token"]
print(f"pérdida de validación : {perdida:.3f}      (sin entrenar sería ln({modelo.cfg.vocab}) = {math.log(modelo.cfg.vocab):.3f})")
print(f"perplejidad           : {math.exp(perdida):.2f}")
print(f"bits por carácter     : {perdida / math.log(2) / caracteres_por_token:.3f}")

# %% [markdown]
# * **Perplejidad** = e^pérdida. Se lee así: «de media, el modelo duda como si tuviera que
#   elegir entre *tantos* tokens igual de probables». Sin entrenar dudaba entre 1.024. Es la
#   medida de la columna *PPL* en la Tabla 3 del artículo.
# * **Bits por carácter**: la pérdida, repartida entre los caracteres. Sirve para comparar
#   modelos con **tokenizadores distintos**: con tokens más grandes hay menos decisiones que
#   tomar y la pérdida por token sube sin que el modelo sea peor. Por carácter, sí es comparable.
#
# ## 2. ¿Dónde duda el modelo?
#
# La pérdida media esconde mucho. Veamos, token a token, qué probabilidad le dio el modelo a
# lo que de verdad venía:

# %%
frase = "Había una vez una gata curiosa que se llamaba Luna. Luna vivía en el bosque. A Luna le gustaba mucho nadar."
ids = [tok.id("<|inicio|>")] + tok.codificar_texto(frase)
x = torch.tensor([ids], device=DISPOSITIVO)
logits, atenciones = modelo.mirar(x)
probabilidades = logits[0].softmax(-1)
print(f"{'token':<12} {'prob.':>7}   sorpresa")
for i in range(1, len(ids)):
    p = probabilidades[i - 1, ids[i]].item()
    print(f"{tok.texto_de(ids[i])!r:<12} {p:>7.1%}   {'█' * round(-math.log(max(p, 1e-9)) * 3)}")

# %% [markdown]
# Mira el nombre. La **primera** vez que aparece «Luna» es imposible de adivinar (podría ser
# cualquier nombre): mucha sorpresa. La segunda y la tercera vez, casi ninguna: el modelo
# **sabe** que toca repetir el nombre que vio antes. Lo mismo con el lugar, la actividad...
# La pérdida que queda es, sobre todo, lo que de verdad es impredecible.
#
# ## 3. Mirar la atención
#
# ¿Cómo sabe qué nombre repetir? Mirando atrás. Para cada capa y cada cabeza, veamos a qué
# token mira más el último «A» (justo antes de que tenga que escribir « Luna»):

# %%
piezas = [tok.texto_de(i) for i in ids]
posicion = max(i for i, p in enumerate(piezas) if p.strip() == "A")
print(f"El token {piezas[posicion]!r} (posición {posicion}) tiene que predecir {piezas[posicion + 1]!r}. ¿A qué mira?\n")
mejor = (0.0, 0, 0)
for capa, pesos in enumerate(atenciones):
    for cabeza in range(pesos.shape[1]):
        fila = pesos[0, cabeza, posicion, : posicion + 1]
        j = int(fila.argmax())
        marca = ""
        if "una" in piezas[j] or "Luna" in piezas[j] or piezas[j].strip() == "L":
            marca = "  <- mira al nombre"
            if fila[j].item() > mejor[0]:
                mejor = (fila[j].item(), capa, cabeza)
        print(f"capa {capa + 1} · cabeza {cabeza + 1}: mira sobre todo a {piezas[j]!r:<10} (posición {j:>2}, {fila[j].item():.0%}){marca}")

# %%
_, capa, cabeza = mejor
taller.ver_atencion(atenciones[capa][0, cabeza], piezas, f"Capa {capa + 1}, cabeza {cabeza + 1}")

# %% [markdown]
# Cada fila del mapa es un token «preguntando»; el color dice a quién mira. El triángulo de
# arriba a la derecha está vacío: es el futuro, tapado por la máscara causal.
#
# Nadie le dijo al modelo «cuando toque escribir el nombre, busca el nombre anterior».
# Ese comportamiento **apareció solo**, porque era la forma de bajar la pérdida. Las Figuras
# 3 a 5 del artículo muestran lo mismo en su modelo de traducción: cabezas que siguen
# dependencias lejanas o resuelven a quién se refiere un pronombre.
#
# > Un aviso honesto: un mapa de atención enseña *adónde mira* una cabeza, no demuestra *por
# > qué* el modelo decide algo. Es una pista, no una explicación completa.
#
# ## 4. Vecinos en la tabla de embeddings
#
# En la clase 3 dijimos que, al entrenar, los tokens que se usan de forma parecida acaban con
# vectores parecidos. Comprobémoslo con la similitud del coseno («¿apuntan dos flechas hacia
# el mismo sitio?»):

# %%
tabla = torch.nn.functional.normalize(modelo.embeddings.weight.float(), dim=-1)
for palabra in [" gato", " bosque", " triste", " Martín", " una"]:
    trozos = tok.codificar_texto(palabra)
    if len(trozos) != 1:
        continue                                       # solo palabras que sean un único token
    parecido = tabla @ tabla[trozos[0]]
    vecinos = [tok.texto_de(i) for i in parecido.topk(7).indices.tolist()[1:]]
    print(f"{palabra!r:<10} -> {vecinos}")

# %% [markdown]
# Los vecinos de un animal suelen ser otros animales; los de un lugar, otros lugares; los de
# un nombre, otros nombres. Nadie le dio una lista de animales: los agrupó porque **aparecen
# en los mismos contextos**.
#
# ## 5. Sobreajuste: aprender de memoria
#
# ¿Qué pasa si hay muy pocos datos? Entrenamos el mismo modelo pequeño con 3.000 cuentos y
# con solo 30, y medimos ambos en cuentos que no han visto:

# %%
from cero.config import ConfigModelo
from cero.cuentos import generar_cuentos
from cero.modelo import GPT
from cero.tokenizador import TokenizadorBPE

cuentos = list(generar_cuentos(300 if RAPIDO else 3000, semilla=0))
tok_mini = TokenizadorBPE.entrenar(cuentos, vocab=512)
validacion = taller.tokens_de(list(generar_cuentos(200, semilla=99)), tok_mini)
historias = {}
for cuantos in (30, len(cuentos)):
    torch.manual_seed(0)
    mini = GPT(ConfigModelo(vocab=tok_mini.vocab, contexto=64, d_modelo=64, n_capas=2, n_cabezas=4))
    historia = taller.entrenar_mini(mini, taller.tokens_de(cuentos[:cuantos], tok_mini), validacion, pasos=30 if RAPIDO else 600,
                                    medir_cada=10 if RAPIDO else 100, dispositivo=DISPOSITIVO)
    historias[f"{cuantos} cuentos"] = historia
    print(f"\ncon {cuantos} cuentos")
    for m in historia:
        print(f"   paso {m['paso']:>4} · entrenamiento {m['entreno']:.2f} · validación {m['validacion']:.2f}")
taller.ver_historias(historias, titulo="Pérdida de validación: pocos datos frente a muchos")

# %% [markdown]
# Con 30 cuentos, la pérdida de **entrenamiento** baja y baja (se los está aprendiendo de
# memoria) mientras la de **validación** deja de bajar y empieza a **subir**. Eso es el
# sobreajuste. Con 3.000, las dos bajan juntas.
#
# Por eso se guarda `mejor.pt` según la validación y no según el entrenamiento.
#
# ## 6. Lo que estas medidas no dicen
#
# Una pérdida baja significa que el texto del modelo **se parece** al del corpus. No dice si
# lo que cuenta es verdad, ni si obedece cuando le pides algo, ni si es coherente a lo largo
# de un texto. Para eso hay que hacerle preguntas con respuesta comprobable: lo haremos en la clase 11.
#
# ## Resumen
#
# * Se evalúa siempre con datos que el modelo no ha visto.
# * Perplejidad = e^pérdida; bits por carácter para comparar tokenizadores.
# * La pérdida se concentra en lo impredecible (la primera mención de un nombre).
# * Si la validación sube mientras el entrenamiento baja, el modelo está memorizando.
#
# ## Ejercicios
#
# 1. Cambia la frase de la sección 2 por otra con otro nombre y otro animal. ¿Sigue habiendo
#    una cabeza que mira al nombre? ¿Es la misma?
# 2. Calcula la pérdida de esa frase: la media de `−ln(p)` sobre sus tokens. ¿Es mayor o
#    menor que la de validación? ¿Por qué?
# 3. Escribe una frase que no tenga nada que ver con cuentos y mira su sorpresa token a token.
# 4. En la sección 5, prueba con 300 cuentos. ¿En qué paso empieza a separarse la validación?
# 5. Abre `cero lab`, ve a «Atención» y explora las cuatro capas. ¿Cambia lo que miran las
#    primeras capas respecto a las últimas?
#
# **Siguiente:** [Clase 9 · Generar](09_generar.ipynb)
