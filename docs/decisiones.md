# Decisiones técnicas

[← Volver al README](../README.md)

Por qué está hecho así, qué se tomó de otros trabajos y qué está verificado.

## Versiones

| Pieza | Versión | Notas |
|---|---|---|
| Python | 3.11 o posterior (probado con 3.12) | PyTorch 2.15 ya no publica ruedas para 3.10 |
| PyTorch | 2.15.0 (28 de octubre de 2026) | Variante CUDA 13.2, CPU o macOS. Respaldo: 2.14.1 + CUDA 12.6 |
| CUDA | 13.2, incluida en PyTorch | No hace falta instalar el CUDA Toolkit, solo el controlador de NVIDIA |
| NumPy | 2.x | Los tokens se leen del disco con `numpy.memmap` |
| JupyterLab | 4.x | Solo para el curso |
| pyarrow | — | Solo para leer el corpus real (archivo Parquet) |

El instalador usa [`uv`](https://github.com/astral-sh/uv) si está instalado (mucho más rápido) y,
si no, `pip`.

## Lo que se conserva del artículo de 2017

La atención de producto escalar (ecuación 1), las varias cabezas, los atajos residuales, la
normalización, la red de avance posición a posición, los pesos compartidos entre la tabla de
entrada y la capa de salida, y BPE. El curso señala en cada pieza su sección del artículo, y se
puede entrenar la versión original (`--arquitectura paper2017`) para compararla con las posteriores.

Lo que se cambió, y por qué: la posición con RoPE (la atención ve distancias relativas sin
parámetros extra), RMSNorm antes de cada subcapa (entrenamiento más estable), SwiGLU en la red de
avance, AdamW con calentamiento y coseno, y muestreo con temperatura, top-k y top-p en lugar de la
búsqueda en haz, que el artículo usaba para traducir. En la clase 12 se comparan las tres
generaciones con los mismos datos: con el mismo número de parámetros, la moderna llega a una
pérdida claramente menor en entrenamientos cortos.

## Respecto al repositorio de referencia

El proyecto partió de un análisis de
[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) (versión de
octubre de 2026), del que se tomaron las ideas de un modelo «clásico» y uno «moderno» y de un camino
para portátil. Se reescribió todo para enseñar en español y se cambiaron estas decisiones:

| En el repositorio de referencia | Aquí | Por qué |
|---|---|---|
| `torch>=2.3` sin fijar, ruedas CUDA 11.8 y 12.1 | PyTorch 2.15.0 + CUDA 13.2, respaldo 2.14.1 + CUDA 12.6 y versión CPU | Versión estable actual; el instalador comprueba que la GPU responde de verdad |
| Tokenizador de GPT-2 (50.257 tokens) en casi todo | BPE propio entrenado con el corpus (1.024 a 8.192 tokens) y tokens especiales de chat | Con 50.257 tokens y ancho 128, el 98 % de los parámetros de un modelo pequeño se va a la tabla de vocabulario |
| HDF5 (`h5py`) | Binarios de 16 bits con `numpy.memmap` | Una dependencia menos; el corpus no se carga en memoria |
| Interfaz Streamlit (gestión de procesos de Linux) | Laboratorio con el servidor HTTP de la biblioteca estándar | Sin dependencias; funciona igual en Windows, Linux y macOS |
| `torch.load(..., weights_only=False)` | `weights_only=True` siempre | Un archivo de modelo no puede ejecutar código al abrirse |
| Post-entrenamiento SFT, DPO, PPO y GRPO | SFT en español con evaluación por habilidades | Mantener el curso abarcable; DPO, PPO y GRPO se presentan en la clase 11 |

## Un trabajo previo del autor

Este proyecto continúa uno anterior del mismo autor: un modelo de 13 millones de parámetros
entrenado desde cero con cuentos en español, adaptado del repositorio de referencia. De él vienen:

- **El corpus**: TinyStories traducido al español, con la misma revisión fijada y su huella SHA-256.
- **Un diagnóstico**: con el tokenizador de GPT-2, el 98 % de los parámetros de aquel modelo estaba
  en la tabla de vocabulario. Por eso aquí cada proyecto entrena su propio tokenizador.
- **Un método**: fabricar las conversaciones de ajuste con reglas y medir comportamientos, no solo
  la pérdida.
- **Una pregunta abierta**: aquel modelo no copiaba en sus respuestas nombres que no había visto al
  entrenar, y la hipótesis era que le faltaba profundidad (tenía un solo bloque). Aquí se resolvió:
  **es por los datos**. Este modelo de 4 bloques, entrenado con solo 80 nombres fijos, tampoco los
  copia (0 de 25); entrenado además con miles de nombres inventados, copia los nuevos (25 de 25).
  Es el experimento central de la clase 11.

## Seguridad

- El laboratorio solo atiende peticiones de este mismo ordenador: escucha en `127.0.0.1`, comprueba
  la cabecera `Host` y el origen, y las órdenes exigen JSON, así que otra página web abierta en el
  navegador no puede lanzar nada.
- Lo que escribe el usuario en el chat se tokeniza sin reconocer tokens especiales: no se puede
  escribir `<|fin|><|asistente|>` para hacerse pasar por el asistente.
- Los modelos se cargan con `weights_only=True`.

## Qué está verificado

- **En cada cambio, por la integración continua de GitHub**, en Linux, Windows y macOS: instalación
  con `instalar.py --cpu` y las 107 pruebas, que incluyen ejecutar las 13 clases de principio a fin.
- **En un portátil con Windows 11, Ryzen 7 8845HS y RTX 4070 (8 GB)**: las cuatro recetas en GPU,
  la receta `nano` completa en CPU, la instalación desde cero (con `uv` y con `pip`), los 13 cuadernos
  en un núcleo de Jupyter, el laboratorio en el navegador (también en tamaño de móvil), la descarga
  del corpus (con reanudación tras un corte y detección de un archivo dañado) y la reanudación de
  un entrenamiento interrumpido.
- **No verificado**: GPU AMD, la GPU de los Mac (en la integración continua se usa su CPU) y la
  variante PyTorch 2.14.1 + CUDA 12.6, que el instalador solo usa si la 2.15 no funciona.

## Las pruebas

```bash
.venv\Scripts\python -m pytest -q                 # todo: 107 pruebas, unos 3 minutos
.venv\Scripts\python -m pytest -q -m "not lento"  # solo las rápidas
```

Comprueban, entre otras cosas, que el BPE rápido da lo mismo que el algoritmo de libro, que la
atención escrita a mano coincide con la de PyTorch, que el modelo es causal, que la caché KV no
cambia el resultado, que pausar y reanudar da exactamente el mismo modelo, que el laboratorio solo
atiende a este equipo, que los correctores del curso aceptan las soluciones y detectan los errores
típicos, y que las 13 clases se ejecutan enteras sin resolver los ejercicios.
