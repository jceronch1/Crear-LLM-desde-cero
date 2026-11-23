# El curso: un LLM y su chatbot, paso a paso

Trece clases en cuadernos de Jupyter. Cada una explica una pieza, la construye a mano, la
compara con la del proyecto y la usa sobre un modelo de verdad. Al terminar, el estudiante ha
entrenado su propio LLM, lo ha convertido en chatbot y ha medido qué aprendió y qué no.

Para abrirlas: `.\cero curso` en Windows o `./cero.sh curso` en Linux y macOS (abre JupyterLab en
esta carpeta), o la opción 5 de `menu.cmd`. También se pueden abrir con VS Code eligiendo como núcleo
el Python de `.venv`.

| Nº | Clase | Duración | Idea central | Artículo |
|---:|---|---:|---|---|
| 0 | [Qué vamos a construir](00_bienvenida.ipynb) | 30 min | El recorrido entero, en miniatura y en un minuto | Resumen y §3 |
| 1 | [El tokenizador](01_tokenizador.ipynb) | 50 min | BPE: fusionar la pareja más frecuente, una y otra vez | §5.1 |
| 2 | [Los datos](02_datos.ipynb) | 40 min | La respuesta está en el propio texto, un lugar a la derecha | Fig. 1 |
| 3 | [Embeddings y posición](03_embeddings_y_posicion.ipynb) | 45 min | Una fila por token; senos, tabla aprendida o RoPE | §3.4, §3.5 |
| 4 | [La atención](04_atencion.ipynb) | 60 min | softmax(QKᵀ/√d)·V, la máscara y las cabezas | §3.2, ec. (1) |
| 5 | [El bloque Transformer](05_bloque.ipynb) | 45 min | Atajos, normalización y red de avance | §3.1, §3.3 |
| 6 | [El modelo completo](06_modelo.ipynb) | 40 min | Logits, pesos atados y la pérdida ln(V) | §3.4, Tabla 3 |
| 7 | [Entrenar](07_entrenar.ipynb) | 60 min | Los cuatro pasos y la tasa de aprendizaje; entrena el modelo | §5.3 |
| 8 | [Evaluar y mirar por dentro](08_evaluar.ipynb) | 50 min | Perplejidad, mapas de atención y sobreajuste | Figs. 3-5 |
| 9 | [Generar](09_generar.ipynb) | 45 min | Temperatura, top-k, top-p y la caché KV | §6.1 |
| 10 | [De LLM a chatbot](10_chat.ipynb) | 60 min | Plantilla, máscara y ajuste fino; lo convierte en chatbot | — |
| 11 | [¿Qué aprendió de verdad?](11_que_aprendio.ipynb) | 50 min | Copiar frente a memorizar, con preguntas nunca vistas | — |
| 12 | [Experimentos y escala](12_experimentos.ipynb) | 60 min + proyecto | Tres arquitecturas, tamaños, corpus reales y proyecto final | Tabla 3 |

Unas **11 horas** de clase. El reparto en sesiones y las soluciones están en la
[guía del profesor](GUIA_DEL_PROFESOR.md).

## Cómo están hechas

* **Se pueden abrir en cualquier orden.** Si una clase necesita el modelo entrenado y no está,
  lo entrena ella misma (unos minutos, una sola vez). Todas usan el mismo proyecto, `curso`,
  que también se puede preparar de antemano desde la consola: `cero todo curso --receta nano`.
* **Ejercicios «Tu turno».** Once funciones para completar (contar parejas, la máscara, la
  atención, RMSNorm, la plantilla de chat...). Cada una tiene un corrector que dice si está
  bien y, si no, da una pista concreta. Sin resolverlos, la clase se sigue ejecutando entera.
* **Ejercicios al final de cada clase**, para casa o para debatir.
* **CPU o GPU.** Todo funciona en los dos. La GPU solo acelera los entrenamientos de las
  clases 7 y 10 (de ~5 minutos a ~1).

## Para el profesor

Las clases se escriben en `fuentes/*.py` (Python normal con marcas `# %%`) y los cuadernos
se generan con `python curso/construir.py`. Así se pueden revisar con cualquier editor y las
pruebas automáticas las ejecutan todas (`pytest -m lento tests/test_curso.py`). Si cambias
una clase, edita la fuente y vuelve a generar el cuaderno.

Las soluciones de los «Tu turno» están en [soluciones.py](soluciones.py).
