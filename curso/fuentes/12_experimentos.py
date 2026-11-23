# %% [markdown]
# # Clase 12 · Experimentos y escala
#
# Ya conoces todas las piezas. En esta última clase las tocas tú: comparar las tres
# generaciones del Transformer, cambiar el tamaño, entrenar con cuentos reales o con tus
# propios textos. Termina con propuestas de proyecto final.
#
# | | |
# |---|---|
# | **Duración** | 60 minutos + proyecto |
# | **En el artículo** | Tabla 3: el artículo hace exactamente esto, cambiar una cosa y medir |
# | **En el código** | `cero/config.py`: recetas y arquitecturas |
# | **Al terminar sabrás** | Diseñar un experimento justo y estimar qué cuesta escalar un modelo |

# %%
import torch

from cero import dispositivo, taller
from cero.config import ARQUITECTURAS, ConfigModelo
from cero.cuentos import generar_cuentos
from cero.modelo import GPT
from cero.taller import RAPIDO
from cero.tokenizador import TokenizadorBPE

DISPOSITIVO = dispositivo.elegir("auto")
cuentos = list(generar_cuentos(400 if RAPIDO else 4000, semilla=0))
tok = TokenizadorBPE.entrenar(cuentos, vocab=512)
tokens = taller.tokens_de(cuentos, tok)
corte = int(0.95 * len(tokens))
entreno, validacion = tokens[:corte], tokens[corte:]
PASOS = 30 if RAPIDO else 400
print(f"{len(tokens):,} tokens · {PASOS} pasos por experimento · {dispositivo.resumen(DISPOSITIVO)}")

# %% [markdown]
# ## 1. Las reglas de un experimento justo
#
# 1. **Cambiar una sola cosa** cada vez.
# 2. Mismos datos, mismos pasos, misma semilla.
# 3. Medir en **validación**.
# 4. Mirar también las muestras, no solo el número.
#
# La Tabla 3 del artículo es eso: el modelo base y, en cada fila, una sola cosa cambiada
# (número de cabezas, tamaño, dropout, tipo de posición...).
#
# ## 2. Tres generaciones del Transformer, con los mismos datos

# %%
historias, tamanos = {}, {}
for nombre, interruptores in ARQUITECTURAS.items():
    torch.manual_seed(0)
    modelo = GPT(ConfigModelo(vocab=tok.vocab, contexto=128, d_modelo=96, n_capas=3, n_cabezas=4, **interruptores))
    tamanos[nombre] = modelo.n_parametros()
    historias[nombre] = taller.entrenar_mini(modelo, entreno, validacion, pasos=PASOS, lr=2e-3,
                                             medir_cada=max(10, PASOS // 8), dispositivo=DISPOSITIVO)
    print(f"{nombre:<10} {tamanos[nombre]:>9,} parámetros · validación final {historias[nombre][-1]['validacion']:.3f}")
taller.ver_historias(historias, titulo="Las tres arquitecturas, mismos datos y mismos pasos")

# %% [markdown]
# Con casi los mismos parámetros, la diferencia es grande. Al preparar el curso lo repetimos
# con dos semillas más y dos tasas de aprendizaje, y quitando el dropout al de 2017: el orden
# se mantuvo siempre (moderna ≈ 0,58, gpt2 ≈ 1,10 y paper2017 ≈ 2,4-2,6 con lr = 0,002).
#
# ¿De dónde sale? En un entrenamiento tan corto pesa mucho **aprender a usar la posición**.
# Con RoPE la atención ve distancias relativas desde el primer paso («el token de hace 3
# lugares»), algo muy útil para copiar el nombre del protagonista. Con posiciones aprendidas,
# el modelo tiene que descubrir primero qué significa cada fila de la tabla de posiciones; y el
# de 2017 suma además la post-norma, que hace más difícil el entrenamiento.
#
# Ojo con generalizar: es **un** experimento, con modelos diminutos, pocos pasos y la misma
# tasa para todos. Con entrenamientos largos las diferencias se estrechan mucho, aunque la
# dirección es la misma que encontraron quienes adoptaron estas piezas.
#
# ## 3. ¿Más grande es mejor?

# %%
historias = {}
for d_modelo, capas in ((32, 2), (64, 2), (128, 4)):
    torch.manual_seed(0)
    modelo = GPT(ConfigModelo(vocab=tok.vocab, contexto=128, d_modelo=d_modelo, n_capas=capas, n_cabezas=4))
    etiqueta = f"{modelo.n_parametros() / 1e3:,.0f} mil parámetros"
    historias[etiqueta] = taller.entrenar_mini(modelo, entreno, validacion, pasos=PASOS, lr=2e-3,
                                               medir_cada=max(10, PASOS // 8), dispositivo=DISPOSITIVO)
    print(f"{etiqueta:<24} validación final {historias[etiqueta][-1]['validacion']:.3f}")
taller.ver_historias(historias, titulo="Mismos datos y pasos, tres tamaños")

# %% [markdown]
# Con los mismos pasos, el modelo grande aprende más **de cada token** que ve. Pero cada paso
# le cuesta más. Las *leyes de escala* estudian ese equilibrio: para un presupuesto de
# cálculo fijo, ¿cuántos parámetros y cuántos tokens? Una regla práctica muy citada
# (Chinchilla, 2022) es entrenar con unos **20 tokens por parámetro**.

# %%
for nombre, parametros, tokens_vistos in [("nano (este curso)", 0.92e6, 3.1e6), ("micro", 5.8e6, 24.6e6),
                                          ("mini", 15.7e6, 82e6), ("base", 35.7e6, 115e6)]:
    print(f"{nombre:<18} {parametros / 1e6:5.1f} M parámetros · {tokens_vistos / 1e6:6.1f} M tokens de preentrenamiento"
          f" · {tokens_vistos / parametros:4.1f} tokens por parámetro")

# %% [markdown]
# Todas nuestras recetas se quedan por debajo de 20: están pensadas para caber en una clase,
# no para exprimir cada parámetro. Subir los pasos es el primer experimento que conviene hacer
# si tienes tiempo de máquina.
#
# ## 4. Cuentos reales
#
# El mundo sintético es pequeño a propósito. Las recetas `micro`, `mini` y `base` usan
# **TinyStories en español**: 265.000 cuentos infantiles reales (traducidos automáticamente),
# con un vocabulario mucho más rico. La primera vez descargan 258 MB.
#
# ```
# cero todo --receta micro        # ~6 M de parámetros
# cero todo --receta mini         # ~16 M, solo GPU
# cero todo --receta base         # ~36 M, solo GPU con 8 GB
# ```
#
# Lo que obtuvimos en un portátil con una RTX 4070 (8 GB):
#
# | Receta | Parámetros | Tiempo total en GPU | Pérdida de validación | Bits por carácter |
# |---|---:|---:|---:|---:|
# | `nano` (sintético) | 0,9 M | ~2 min | 0,42 | 0,17 |
# | `micro` | 5,8 M | ~3 min | 2,02 | 0,75 |
# | `mini` | 15,7 M | ~19 min | 1,69 | 0,63 |
# | `base` | 35,7 M | ~50 min | 1,60 | 0,55 |
#
# > La pérdida de `nano` es mucho más baja porque su mundo es muchísimo más previsible, no
# > porque el modelo sea mejor. Entre corpus distintos **no se pueden comparar** las pérdidas.
# > Entre `micro`, `mini` y `base` la pérdida tampoco es comparable (cada una tiene su propio
# > tokenizador), pero los **bits por carácter** sí: 0,75 → 0,63 → 0,55. Más grande, mejor.
#
# Una muestra del modelo `micro` (sin retocar), tras 3 minutos de entrenamiento:
#
# > *Había una vez un granjero regordete que estaba buscando una comida deliciosa. Estaba muy
# > feliz y le encantaban los días. Un día, vio un arbusto de...*
#
# Gramática correcta, vocabulario variado y frases con sentido local; la historia todavía no
# va a ninguna parte. Si en tu equipo hay GPU, lanza `micro` y compara su chat con el de `nano`.
#
# ## 5. Tus propios textos
#
# Cualquier carpeta de archivos `.txt` sirve de corpus:
#
# ```
# cero nuevo poemas --receta nano --texto C:\ruta\a\mis_textos
# cero todo ...                    (o paso a paso: cero datos, cero entrenar, cero generar)
# ```
#
# Con pocos datos, vigila la curva de validación: si sube mientras baja la de entrenamiento,
# el modelo está memorizando (clase 8). Un libro entero son unos 500.000 caracteres: alcanza
# para un modelo `nano` y para ver el sobreajuste en directo.
#
# ## 6. Cambiar cualquier ajuste
#
# Cada proyecto guarda su receta en `proyectos/<nombre>/config.json`. Se puede editar antes
# de cada etapa:
#
# | Si cambias... | Hay que repetir... |
# |---|---|
# | `vocab`, el corpus | todo: `cero datos`, `cero entrenar`, `cero afinar` |
# | `modelo` (capas, ancho, interruptores) | `cero entrenar` y `cero afinar` |
# | `entreno` (pasos, lr, lote) | `cero entrenar` y `cero afinar` |
# | `chat` | solo `cero afinar` |
#
# Para comparar, mejor un proyecto por variante (`cero nuevo variante-a ...`) que sobrescribir.
#
# ## 7. Propuestas de proyecto final
#
# Elige una, plantea una **hipótesis antes de ejecutar** y defiende el resultado con números
# y muestras sin retocar:
#
# 1. **Arquitecturas.** Entrena la receta `nano` completa con `paper2017`, `gpt2` y `moderna`.
#    Compara pérdida y la tabla de `cero evaluar`. ¿Alguna aprende antes a copiar nombres?
# 2. **Vocabulario.** `nano` con vocabulario 512, 1.024 y 2.048. Compara en **bits por
#    carácter**, no en pérdida (clase 8). ¿Qué tamaño es mejor para este mundo?
# 3. **Profundidad.** 2, 4 y 8 capas con el mismo número de parámetros (ajusta el ancho). ¿Con
#    cuántas capas aparece la habilidad de copiar nombres nunca vistos?
# 4. **Enseñar a sumar.** Cambia las respuestas de suma en `cero/chat.py` para que muestren el
#    cálculo («7 + 5: 7, 8, 9, 10, 11, 12. Son 12.»). ¿Mejoran las sumas nunca vistas?
# 5. **Un corpus propio.** Entrena con los textos de una asignatura, un autor o un tema. ¿En
#    qué punto empieza a memorizar? ¿Cómo lo detectas?
# 6. **Un chatbot con otra personalidad.** Cambia `CHARLA` y `NO_SE` en `cero/chat.py` para
#    que el asistente hable de otra manera. ¿Cuántos ejemplos hacen falta para que lo aprenda?
#
# ## 8. Para seguir aprendiendo
#
# * **El artículo**: *Attention Is All You Need* (Vaswani et al., 2017). Después de este curso
#   deberías poder leer las secciones 3 a 6 de principio a fin.
# * **TinyStories** (Eldan y Li, 2023): por qué los cuentos sencillos sirven para estudiar
#   modelos pequeños.
# * **train-llm-from-scratch** (FareedKhan-dev, GitHub): el repositorio que inspiró este
#   proyecto. Añade alineamiento con preferencias (DPO, PPO) y aprendizaje por refuerzo (GRPO).
# * **nanoGPT y nanochat** (Andrej Karpathy, GitHub): las mismas ideas a mayor escala, con
#   técnicas de optimización actuales.
#
# ## Ejercicios
#
# 1. Repite la sección 2 con `PASOS = 2000`. ¿Se estrecha la diferencia entre arquitecturas?
#    Para aislar la causa, crea una variante moderna con `posicion="aprendida"`: ¿a cuál se parece más?
# 2. En la sección 3, ¿cuántos segundos tarda cada tamaño? Calcula «pérdida conseguida por
#    segundo de cálculo». ¿Cuál sale mejor parado?
# 3. Estima: con la velocidad que da `cero comprobar` en tu equipo, ¿cuánto tardaría la receta
#    `nano` en ver 20 tokens por parámetro?
# 4. Elige un proyecto final de la sección 7 y escribe la hipótesis en una frase.
