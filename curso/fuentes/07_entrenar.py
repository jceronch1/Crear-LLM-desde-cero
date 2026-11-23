# %% [markdown]
# # Clase 7 · Entrenar: de ruido a español
#
# Tenemos un modelo con 900.000 números al azar. Entrenar es ir cambiándolos, poco a poco,
# para que la probabilidad que el modelo da al token correcto sea cada vez mayor.
#
# | | |
# |---|---|
# | **Duración** | 60 minutos (incluye unos minutos de entrenamiento real) |
# | **En el artículo** | §5.3 (optimizador y tasa de aprendizaje), §5.4 (regularización) |
# | **En el código** | `cero/entrenar.py` |
# | **Al terminar sabrás** | Los cuatro pasos del bucle, qué es la tasa de aprendizaje y cómo leer una curva de pérdida |

# %% [markdown]
# ## 1. La pérdida, a mano
#
# Para una posición, el modelo da una probabilidad a cada token. La pérdida es
# `−ln(probabilidad del token correcto)`:

# %%
import math

import torch
import torch.nn.functional as F

for p in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99):
    print(f"si al token correcto le da un {p:>6.1%} -> pérdida {-math.log(p):5.2f}")

# %% [markdown]
# Acertar con seguridad cuesta casi 0; fallar con seguridad cuesta muchísimo. La pérdida de
# un lote es la media de todas sus posiciones. PyTorch la calcula con `F.cross_entropy`:

# %%
logits = torch.tensor([[2.0, 0.5, -1.0, 0.1]])          # puntuaciones de 4 tokens posibles
correcto = torch.tensor([0])
probabilidades = logits.softmax(-1)
print("probabilidades:", [round(v, 3) for v in probabilidades[0].tolist()])
print("a mano        :", round(-math.log(probabilidades[0, 0].item()), 4))
print("cross_entropy :", round(F.cross_entropy(logits, correcto).item(), 4))

# %% [markdown]
# ## 2. Los cuatro pasos
#
# ```python
# _, perdida = modelo(x, y)        # 1. HACIA DELANTE: ¿cómo de mal predice?
# perdida.backward()               # 2. HACIA ATRÁS: ¿hacia dónde mover cada peso? (el gradiente)
# clip_grad_norm_(...)             # 3. RECORTAR: si el gradiente es enorme, se acorta
# optimizador.step()               # 4. ACTUALIZAR: mover cada peso un poco
# ```
#
# El paso 2 es la *retropropagación*: PyTorch calcula, para **cada uno** de los 900.000
# números, cuánto cambiaría la pérdida si se moviera un poquito. No lo programamos nosotros.
#
# La prueba clásica de que todo está bien conectado: darle al modelo **un único lote** una y
# otra vez. Si el modelo y el gradiente funcionan, se lo aprende de memoria y la pérdida cae a casi 0.

# %%
from cero import dispositivo, taller
from cero.config import ConfigEntreno, ConfigModelo
from cero.entrenar import crear_optimizador
from cero.modelo import GPT
from cero.taller import RAPIDO

DISPOSITIVO = dispositivo.elegir("auto")
torch.manual_seed(0)
modelo = GPT(ConfigModelo(vocab=1024, contexto=32, d_modelo=64, n_capas=2, n_cabezas=4)).to(DISPOSITIVO)
optimizador = crear_optimizador(modelo, ConfigEntreno(lr=3e-3), DISPOSITIVO)
x = torch.randint(0, 1024, (4, 33), device=DISPOSITIVO)          # un lote fijo de 4 textos inventados
for paso in range(101):
    _, perdida = modelo(x[:, :-1], x[:, 1:])                     # 1
    optimizador.zero_grad()
    perdida.backward()                                           # 2
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)     # 3
    optimizador.step()                                           # 4
    if paso % 20 == 0:
        print(f"paso {paso:>3} · pérdida {perdida.item():.4f}")

# %% [markdown]
# De 6,9 (azar: ln 1024) a casi 0. No ha aprendido nada útil, solo ha memorizado 128 tokens,
# pero demuestra que el mecanismo funciona. Es lo primero que se comprueba al escribir un
# modelo nuevo.
#
# ## 3. La tasa de aprendizaje
#
# ¿Cuánto es «mover un poco»? Lo decide la **tasa de aprendizaje** (*learning rate*), el ajuste
# más delicado de todos: demasiado alta y el entrenamiento explota; demasiado baja y no avanza.
#
# No se usa un valor fijo. El artículo (§5.3, ecuación 3) propuso subirla al principio y
# bajarla después:
#
# $$lr = d_{modelo}^{-0.5} \cdot \min(paso^{-0.5},\; paso \cdot calentamiento^{-1.5})$$
#
# Hoy se usa la misma idea con otra forma: **calentamiento lineal y descenso en coseno**.

# %%
def tasa_del_articulo(paso, d_modelo=128, calentamiento=100):
    paso = max(paso, 1)
    return d_modelo ** -0.5 * min(paso ** -0.5, paso * calentamiento ** -1.5)


cfg = ConfigEntreno(pasos=1000, lr=3e-3, calentamiento=100, lr_final=0.1)
from cero.entrenar import tasa_de_aprendizaje

print(f"{'paso':>5} {'artículo (2017)':>17} {'calentamiento + coseno':>24}")
for paso in (0, 25, 50, 100, 250, 500, 750, 999):
    print(f"{paso:>5} {tasa_del_articulo(paso):>17.5f} {tasa_de_aprendizaje(paso, cfg):>24.5f}")

# %% [markdown]
# Las dos **empiezan con cuidado** (al principio los pesos son ruido y los gradientes, poco
# fiables), **aceleran** y **frenan** al final para afinar.
#
# ### Tu turno · calentamiento y coseno
# Con `lr = 0.001`, `calentamiento = 100`, `pasos = 1000` y `lr_final = 0.1`:
#
# * durante el calentamiento (`paso < 100`) la tasa sube en línea recta: `lr · (paso + 1) / 100`
# * después baja siguiendo medio coseno desde `lr` hasta `lr · lr_final`

# %%
def mi_tasa(paso, lr=1e-3, calentamiento=100, pasos=1000, lr_final=0.1):
    if paso < calentamiento:
        raise NotImplementedError                    # TU CÓDIGO: la subida en línea recta
    avance = (paso - calentamiento) / (pasos - calentamiento)        # de 0 a 1
    coseno = 0.5 * (1 + math.cos(math.pi * avance))                  # de 1 a 0
    raise NotImplementedError                        # TU CÓDIGO: lr · (lr_final + (1 − lr_final) · coseno)


taller.comprobar_tasa(mi_tasa)

# %% [markdown]
# ### Un experimento: tres tasas de aprendizaje
#
# Mismo modelo, mismos datos, mismos pasos. Solo cambia la tasa.

# %%
from cero.cuentos import generar_cuentos
from cero.tokenizador import TokenizadorBPE

cuentos = list(generar_cuentos(300 if RAPIDO else 3000, semilla=0))
tok_mini = TokenizadorBPE.entrenar(cuentos, vocab=512)
tokens = taller.tokens_de(cuentos, tok_mini)
corte = int(0.95 * len(tokens))
historias = {}
for lr in (3e-5, 3e-3, 3e-1):
    torch.manual_seed(0)
    mini = GPT(ConfigModelo(vocab=tok_mini.vocab, contexto=64, d_modelo=64, n_capas=2, n_cabezas=4))
    historias[f"lr = {lr:g}"] = taller.entrenar_mini(mini, tokens[:corte], tokens[corte:], pasos=20 if RAPIDO else 200,
                                                     lr=lr, medir_cada=10 if RAPIDO else 40, dispositivo=DISPOSITIVO)
    print(f"lr = {lr:<7g} validación: " + "  ".join(f"{m['validacion']:.2f}" for m in historias[f"lr = {lr:g}"]))
taller.ver_historias(historias, titulo="La misma receta con tres tasas de aprendizaje")

# %% [markdown]
# Con 0,00003 apenas se mueve; con 0,3 se queda atascado muy arriba (da saltos demasiado
# grandes); con 0,003 aprende bien. Encontrar la zona buena es cuestión de probar.
#
# ¿Por qué con 0,3 no «explota» del todo? Por dos protecciones del bucle: el **recorte del
# gradiente** y **AdamW**, que mide cada paso en relación con el tamaño habitual del gradiente.
# Sin ellas, una tasa así convierte la pérdida en `nan` («no es un número») en pocos pasos.
#
# ## 4. El entrenamiento de verdad
#
# Ahora el modelo del proyecto: 900.000 parámetros y cuatro millones de tokens de cuentos.
# Es lo mismo que la orden `cero entrenar`. **Tarda unos minutos**; mientras, lee lo que
# imprime: cada cierto número de pasos mide la validación y escribe una muestra, siempre con
# el mismo comienzo y la misma semilla. Verás al modelo aprender a escribir.
#
# (Si el modelo ya estaba entrenado, la celda no repite el trabajo.)

# %%
proyecto = taller.proyecto_del_curso()
taller.asegurar(proyecto, "base", DISPOSITIVO)
print(proyecto.estado()["base"])

# %%
from cero.entrenar import leer_metricas

eventos = leer_metricas(proyecto.etapa("base"))
taller.ver_curvas(eventos, vocab=proyecto.config["modelo"]["vocab"], titulo="Preentrenamiento del proyecto del curso")
for e in eventos:
    if e["tipo"] == "eval":
        print(f"paso {e['paso']:>5} · validación {e['perdida_val']:.3f} · «{' '.join((e['muestra'] or '').split())[:105]}»")

# %% [markdown]
# Lee las muestras de arriba abajo. Lo normal es ver tres etapas:
#
# 1. **Paso 0**: trozos al azar. Pérdida ≈ ln(1024) = 6,93.
# 2. **Primeros cientos de pasos**: la pérdida se desploma. Ya son frases gramaticales y
#    concuerdan («una vaca tímida»), pero el cuento cambia de protagonista en cada frase.
# 3. **Más adelante**, con la pérdida ya baja, hay un segundo escalón más pequeño: el nombre
#    del protagonista **se mantiene** de una frase a otra. El modelo ha aprendido a usar la
#    atención para mirar atrás y copiar el nombre. Es exactamente lo que vimos en la clase 4.
#
# ### Lo que hace el proyecto además del bucle
#
# | | |
# |---|---|
# | **AdamW** | El optimizador: adapta el tamaño del paso a cada peso (el artículo usó Adam, su antecesor) |
# | **Recorte del gradiente** | Si un lote da un gradiente enorme, se acorta para no dar un salto al vacío |
# | **Precisión mixta** | En GPU, las cuentas intermedias en 16 bits: la mitad de memoria y más velocidad |
# | **Validación periódica** | Siempre con los mismos lotes, para que las mediciones sean comparables |
# | **Puntos de control** | `mejor.pt` (el mejor en validación) y `ultimo.pt` (para reanudar tras Ctrl+C) |
#
# ## Resumen
#
# * Pérdida = −ln(probabilidad del token correcto). Empieza en ln(V).
# * Cada paso: hacia delante, hacia atrás, recortar, actualizar.
# * La tasa de aprendizaje sube, se mantiene alta y baja: es el ajuste que más importa.
# * La curva de validación y las muestras cuentan lo que el modelo va aprendiendo.
#
# ## Ejercicios
#
# 1. En el experimento de la sección 3, prueba con `lr = 3e-2` y `lr = 3e-4`. ¿Dónde está el límite?
# 2. En la sección 2, quita `optimizador.zero_grad()`. ¿Qué pasa y por qué? (Los gradientes
#    se **suman** a los de la vuelta anterior.)
# 3. ¿En qué paso de tu entrenamiento empieza el modelo a mantener el nombre del protagonista?
#    Compáralo con el de un compañero: ¿coincide? (Con la misma semilla, debería.)
# 4. Desde la consola: `cero nuevo prueba --receta nano`, edita `proyectos/prueba/config.json`
#    para poner `"lr": 0.05` y lanza `cero entrenar -p prueba --pasos 500`. Compara la
#    validación final y las muestras con las del proyecto del curso.
# 5. Lanza `cero entrenar -p prueba` con la tasa buena, pulsa Ctrl+C a la mitad y continúa
#    con `cero entrenar -p prueba --reanudar`. ¿Da un salto la pérdida al reanudar?
#
# **Siguiente:** [Clase 8 · Evaluar y mirar por dentro](08_evaluar.ipynb)
