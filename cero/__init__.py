"""cero · un LLM y su chatbot, construidos desde cero para aprender cómo funcionan.

El paquete está pensado para leerse de arriba abajo, en este orden:

    tokenizador.py   texto  ->  números          (BPE a nivel de bytes, escrito a mano)
    datos.py         números ->  lotes (x, y)     (predecir el token siguiente)
    modelo.py        el Transformer decodificador (atención, bloque, GPT)
    entrenar.py      el bucle de entrenamiento    (AdamW, coseno, precisión mixta)
    generar.py       del modelo al texto          (temperatura, top-k, top-p, caché KV)
    chat.py          de LLM a chatbot             (plantilla de chat y ajuste fino)

Cada archivo corresponde a una o dos clases del curso (carpeta `curso/`).
"""

__version__ = "1.0.0"
