# %% [markdown]
# # Clase 0 · Qué vamos a construir
#
# **En este curso vas a construir un modelo de lenguaje (LLM) y su chatbot desde cero**: sin
# descargar ningún modelo ya entrenado y entendiendo cada pieza. Al final tendrás un programa
# que conversa contigo, y sabrás explicar por qué responde lo que responde.
#
# | | |
# |---|---|
# | **Duración** | 30 minutos |
# | **Necesitas** | Haber ejecutado el instalador (`instalar.cmd` o `sh instalar.sh`) |
# | **Al terminar sabrás** | Qué es un LLM por dentro, qué partes tiene y qué haremos en cada clase |
#
# > **Cómo se usa un cuaderno.** Cada bloque gris es código: se ejecuta con `Mayús + Intro`.
# > Ejecútalos en orden, de arriba abajo. Puedes cambiar lo que quieras y volver a ejecutar:
# > estropear cosas a propósito es la mejor forma de entenderlas.

# %% [markdown]
# ## 1. ¿Funciona todo?
#
# El mismo código corre en una **CPU** normal o en una **GPU** (tarjeta gráfica NVIDIA o chip
# de Apple). Solo cambia la velocidad. Veamos qué tiene este equipo.

# %%
from cero import dispositivo

DISPOSITIVO = dispositivo.elegir("auto")          # "auto" elige la GPU si hay; puedes poner "cpu"
print("Aquí se va a calcular en:", dispositivo.resumen(DISPOSITIVO))

# %% [markdown]
# ## 2. La idea entera, en una frase
#
# Un modelo de lenguaje hace **una sola cosa**: dado un texto, calcula qué probabilidad tiene
# cada posible «trozo» de ser el siguiente.
#
# ```
# "Había una vez un gato que se llamaba"  ->   Milo 31 %   Leo 12 %   Luna 0,1 %   mesa 0,0001 % ...
# ```
#
# Escribir es repetir eso muchas veces: sortear un trozo según esas probabilidades, añadirlo
# al texto y volver a preguntar. **Un chatbot es exactamente lo mismo**, con un texto que
# tiene forma de conversación.
#
# Todo lo demás (tokens, atención, entrenamiento...) son los detalles de cómo se consigue
# que esas probabilidades sean buenas.

# %% [markdown]
# ## 3. El recorrido
#
# ```
#  texto ──► tokens ──► lotes (x, y) ──► MODELO ──► pérdida ──► entrenar ──► generar ──► conversar
#  clase 2    clase 1      clase 2      clases 3-6   clase 7     clase 7     clase 9    clases 10-11
# ```
#
# | Clase | Tema | La pregunta que responde | En el código |
# |---:|---|---|---|
# | 1 | Tokenizador | ¿Cómo se convierte un texto en números? | `cero/tokenizador.py` |
# | 2 | Datos y lotes | ¿Qué se le enseña exactamente al modelo? | `cero/datos.py` |
# | 3 | Embeddings y posición | ¿Cómo «significa» algo un número? ¿Y el orden? | `cero/modelo.py` |
# | 4 | Atención | ¿Cómo usa una palabra a las anteriores? | `cero/modelo.py` |
# | 5 | El bloque Transformer | ¿Qué pieza se repite, y por qué funciona apilada? | `cero/modelo.py` |
# | 6 | El modelo completo | ¿Cómo encaja todo? ¿Dónde están los parámetros? | `cero/modelo.py` |
# | 7 | Entrenar | ¿Cómo pasa de ruido a escribir español? | `cero/entrenar.py` |
# | 8 | Evaluar y mirar por dentro | ¿Cómo sé si ha aprendido? ¿Qué ha aprendido? | `cero/entrenar.py` |
# | 9 | Generar | ¿Cómo elige cada palabra? | `cero/generar.py` |
# | 10 | De LLM a chatbot | ¿Qué le falta para conversar? | `cero/chat.py` |
# | 11 | ¿Qué aprendió de verdad? | ¿Entiende, copia o memoriza? | `cero/chat.py` |
# | 12 | Experimentos | ¿Qué cambia si cambio...? ¿Cómo lo hago más grande? | `cero/config.py` |

# %% [markdown]
# ## 4. El artículo que lo empezó todo
#
# En 2017, ocho investigadores de Google publicaron **«Attention Is All You Need»**. Proponían
# una arquitectura nueva, el **Transformer**, que prescindía de las redes recurrentes y lo
# fiaba todo a un mecanismo llamado *atención*. Estaba pensada para traducir, pero todos los
# LLM actuales (GPT, Claude, Llama, Gemini...) descienden de ella.
#
# El Transformer original tiene dos mitades: un **codificador** que lee la frase a traducir y
# un **decodificador** que escribe la traducción. Para *continuar* un texto basta con el
# decodificador, y eso es lo que construiremos: la mitad derecha de su Figura 1.
#
# En estos años se han cambiado algunas piezas. Las iremos viendo; este es el resumen:
#
# | Pieza | Artículo de 2017 | GPT-2 (2019) | Modelos actuales (y este curso) |
# |---|---|---|---|
# | Partes | codificador + decodificador | solo decodificador | solo decodificador |
# | Posición de cada token | senos y cosenos (§3.5) | tabla aprendida | RoPE (rotaciones) |
# | Normalización | LayerNorm, después de sumar | LayerNorm, antes | RMSNorm, antes |
# | Red de avance | ReLU (§3.3) | GELU | SwiGLU |
# | Optimizador | Adam + su propia fórmula (§5.3) | Adam | AdamW + coseno |
# | Para escribir | búsqueda en haz (§6.1) | muestreo | muestreo (temperatura, top-p) |
#
# Lo que **no** ha cambiado es el corazón: la atención de producto escalar (ecuación 1), las
# varias cabezas (§3.2.2), los atajos residuales, compartir la matriz de entrada y la de
# salida (§3.4) y la tokenización BPE (§5.1). En el curso, cada vez que aparezca una de esas
# ideas verás al lado la sección del artículo donde está.

# %% [markdown]
# ## 5. Un adelanto: un LLM diminuto en menos de un minuto
#
# Para que veas adónde vamos, las celdas siguientes hacen **todo el recorrido en miniatura**:
# fabrican unos cuentos, aprenden un tokenizador, crean un modelo de unos 200.000 parámetros,
# lo entrenan unos segundos y le piden que escriba. No hace falta que entiendas ahora cada
# línea: es justo lo que vamos a aprender.

# %%
import torch

from cero.config import ConfigEntreno, ConfigModelo
from cero.cuentos import generar_cuentos
from cero.generar import completar
from cero.modelo import GPT
from cero.taller import RAPIDO
from cero.tokenizador import INICIO, TokenizadorBPE

# 1) El texto: 3.000 cuentos generados por reglas (no se descarga nada).
cuentos = list(generar_cuentos(300 if RAPIDO else 3000, semilla=0))
print(cuentos[0])

# %%
# 2) El tokenizador: aprende 512 trozos de texto y convierte los cuentos en números.
tok = TokenizadorBPE.entrenar(cuentos, vocab=512)
tokens = torch.tensor([i for c in cuentos for i in [tok.id(INICIO)] + tok.codificar_texto(c)])
print(f"{len(tokens):,} tokens. Los 12 primeros: {tokens[:12].tolist()}")
print("Que son:", [tok.texto_de(i) for i in tokens[:12].tolist()])

# %%
# 3) El modelo: un Transformer pequeño, con sus pesos al azar.
torch.manual_seed(0)
cfg = ConfigModelo(vocab=tok.vocab, contexto=128, d_modelo=64, n_capas=2, n_cabezas=4)
modelo = GPT(cfg).to(DISPOSITIVO)
print(f"Modelo de {modelo.n_parametros():,} parámetros")
print("Sin entrenar escribe esto:", repr(completar(modelo, tok, "Había una vez", max_nuevos=25, semilla=1, sin_parar=True)))

# %%
# 4) Entrenar: 300 pasos de «adivina el token siguiente».
from cero.entrenar import crear_optimizador

pasos = 20 if RAPIDO else 300
optimizador = crear_optimizador(modelo, ConfigEntreno(lr=3e-3), DISPOSITIVO)
for paso in range(pasos + 1):
    inicios = torch.randint(0, len(tokens) - 129, (16,))
    x = torch.stack([tokens[i : i + 128] for i in inicios]).to(DISPOSITIVO)
    y = torch.stack([tokens[i + 1 : i + 129] for i in inicios]).to(DISPOSITIVO)
    _, perdida = modelo(x, y)                    # ¿cómo de mal predice el token siguiente?
    optimizador.zero_grad()
    perdida.backward()                           # ¿hacia dónde hay que mover cada peso?
    optimizador.step()                           # moverlos un poco
    if paso % (pasos // 5) == 0:
        print(f"paso {paso:>3} · pérdida {perdida.item():.3f}")

# %%
# 5) Generar: ahora escribe esto.
for semilla in (1, 2):
    print("Había una vez" + completar(modelo, tok, "Había una vez", max_nuevos=60, semilla=semilla), "\n")

# %% [markdown]
# Ha pasado de escribir ruido a escribir algo que **parece** un cuento: palabras de verdad,
# concordancia («una gata», «un perro»), frases que empiezan y acaban. Y solo ha entrenado
# unos segundos.
#
# No sabe nada que no estuviera en esos cuentos, pierde el hilo y todavía no conversa. A lo
# largo del curso lo haremos más grande, lo entrenaremos mejor y lo convertiremos en un chatbot.
#
# ## 6. Las órdenes de la herramienta
#
# Todo lo que hacen los cuadernos se puede hacer también desde la consola con la orden `cero`
# (en Windows se escribe `.\cero`, y en Linux y macOS `./cero.sh`, desde la carpeta del proyecto):
#
# ```
# cero comprobar              ¿cuánto tardará cada receta en este equipo?
# cero todo --receta nano     el recorrido completo: datos, preentrenamiento y ajuste para chat
# cero chat                   conversar con el modelo
# cero evaluar                medir qué aprendió
# cero lab                    el laboratorio: una página para verlo todo por dentro
# ```
#
# ## Para pensar antes de la clase 1
#
# 1. El modelo solo trabaja con números. ¿Cómo convertirías tú un texto en una lista de
#    números? Piensa al menos dos formas y un inconveniente de cada una.
# 2. En el adelanto, la pérdida empezó cerca de 6,2. Con un vocabulario de 512 trozos,
#    ¿qué tiene de especial el número ln(512) = 6,24?
# 3. ¿Por qué crees que entrenamos con cuentos infantiles y no con Wikipedia?
#
# **Siguiente:** [Clase 1 · El tokenizador](01_tokenizador.ipynb)
