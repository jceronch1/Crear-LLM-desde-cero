# Guía del profesor

Lo necesario para dar el curso: cómo preparar el aula, el reparto en sesiones, los momentos
clave de cada clase, las respuestas de los ejercicios y qué hacer cuando algo falla.

---

## 1. Antes de la primera sesión

> En esta guía, `cero ...` se escribe `.\cero ...` en Windows y `./cero.sh ...` en Linux y
> macOS (desde la carpeta del proyecto). En Windows también está el menú de doble clic `menu.cmd`.

**En cada ordenador del aula** (o una semana antes, como deberes):

```
instalar.cmd                 (Windows)       ·       sh instalar.sh      (Linux / macOS)
cero comprobar
```

El instalador detecta si hay GPU NVIDIA: si la hay, instala PyTorch con CUDA (2,1 GB de
descarga); si no, la versión para CPU (130 MB). Comprueba que la GPU funciona de verdad y, si
no, cae a la versión para CPU sin romper nada. `cero comprobar` dice cuánto tardará cada
receta en ese equipo.

**Recomendado: preparar el modelo del curso por adelantado.** Las clases 7 y 10 entrenan el
modelo en directo (≈1 min con GPU, ≈5 min con CPU cada una). Si prefieres no esperar en
clase, déjalo hecho antes:

```
cero todo curso --receta nano
```

Las clases detectan que ya existe y no repiten el trabajo. Si un estudiante borra la carpeta
`proyectos/curso`, la clase que lo necesite lo vuelve a entrenar sola.

**Comprobación de que todo funciona** (unos 3 minutos):

```
.venv\Scripts\python -m pytest -q
```

## 2. Reparto en sesiones

| Sesión | Clases | Duración | Hilo |
|---:|---|---:|---|
| 1 | 0 y 1 | 1 h 20 min | Qué es un LLM y cómo lee: del texto a los números |
| 2 | 2 y 3 | 1 h 25 min | Qué se le enseña y cómo se representa |
| 3 | 4 | 1 h | La atención, a mano |
| 4 | 5 y 6 | 1 h 25 min | El bloque y el modelo completo |
| 5 | 7 | 1 h | Entrenar de verdad |
| 6 | 8 y 9 | 1 h 35 min | Medirlo, mirarlo por dentro y hacerlo escribir |
| 7 | 10 | 1 h | Convertirlo en chatbot |
| 8 | 11 y 12 | 1 h 50 min | ¿Qué aprendió? Experimentos y proyecto final |
| 9 | Proyecto | — | Presentación de los proyectos (sección 7 de la clase 12) |

**Curso corto (unas 5 horas):** 0 · 1 (secciones 1-4) · 2 (secciones 4-5) · 4 · 7 · 10 · 11.
Se pierde la parte de arquitectura detallada, pero se mantiene el relato completo.

## 3. Los momentos que no hay que pasar deprisa

1. **Clase 0, sección 5.** El modelo pasa de escribir ruido a escribir algo parecido a un
   cuento en menos de un minuto. Pide que lo lean en voz alta antes de explicar nada.
2. **Clase 2, sección 4.** La tabla «el modelo ve... y debe predecir». Es cuando se entiende
   que no hace falta etiquetar datos.
3. **Clase 4, paso 2.** Pregunta antes de mostrarla: «¿qué pasaría si al calcular la posición
   2 pudiera mirar la 3?». Que lo descubran ellos.
4. **Clase 7, sección 4.** Las muestras durante el entrenamiento: primero ruido, luego
   gramática, y más tarde (escalón pequeño en la curva) el nombre del protagonista se mantiene.
5. **Clase 8, sección 2.** La sorpresa token a token: la primera «Luna» es imprevisible, la
   segunda no. Es la mejor explicación de qué mide la pérdida.
6. **Clase 10, sección 6.** Misma pregunta al modelo base y al chatbot. Pide una predicción
   antes: casi nadie espera tanto cambio con unos minutos de ajuste y sin tocar la arquitectura.
7. **Clase 11, sección 4.** «3 + 7 = 9». El modelo que parecía saber sumar recitaba una tabla.
   Y, en cambio, sí copia nombres que nunca vio. Discutir por qué es lo más valioso del curso.

## 4. Notas clase por clase

**Clase 0.** Sirve para ajustar expectativas: vamos a construir *la misma máquina* que un
LLM grande, a escala de juguete. Si alguien espera un ChatGPT, mejor aclararlo aquí.

**Clase 1.** Los dos «Tu turno» (contar parejas, fusionar) son la mejor manera de que el
algoritmo se entienda. En la sección 3 las últimas fusiones son trozos que cruzan palabras
(«ata la »): es un fallo provocado a propósito, que motiva el corte en palabras de la sección 4.

**Clase 2.** Insiste en la validación: es la base de todo el rigor posterior (clases 8 y 11).

**Clase 3.** Si el grupo flojea en trigonometría, basta con la analogía del reloj (agujas que
giran a velocidades distintas) y con comprobar numéricamente la propiedad de RoPE.

**Clase 4.** La más densa. Si se atascan, haz el cálculo de la sección 1 en la pizarra con
dos tokens. El ejercicio `mi_atencion` es el más importante del curso: que nadie lo salte.

**Clase 5.** El experimento de los atajos es muy visual: sin atajo el gradiente llega como
10⁻⁸. Pregunta qué significa eso para las primeras capas.

**Clase 6.** Que calculen ln(1024) con la calculadora antes de ejecutar la celda de la pérdida.

**Clase 7.** Mientras entrena (1-5 minutos), pide que anoten en qué paso empieza el modelo a
mantener el nombre del protagonista. Con la misma semilla y el mismo tipo de equipo deberían
ver lo mismo; entre CPU y GPU cambia algo (la GPU calcula con 16 bits), y es un buen tema de debate.

**Clase 8.** El aviso sobre los mapas de atención (una pista, no una explicación) conviene
remarcarlo. El experimento de sobreajuste se entiende mejor dibujando las dos curvas.

**Clase 9.** Muy participativa: que propongan comienzos y temperaturas. La medida de la caché
se aprecia en CPU; en GPU, con este modelo diminuto, apenas se nota (está explicado en la clase).

**Clase 10.** Entrena el ajuste (1-5 minutos). Deja tiempo para conversar con el chatbot en
`cero lab` (vista «Conversar»), con la casilla del modelo base para comparar.

**Clase 11.** Plantéala como un juicio: ¿entiende o no? Cada grupo defiende una postura con
números de la tabla. El experimento de nombres fijos ya está medido en la tabla; repetirlo es opcional.

**Clase 12.** Los experimentos de arquitectura tardan alrededor de un minuto. Si hay GPU en
el aula, lanza `cero todo --receta micro` al principio de la sesión para comparar al final.

## 5. Soluciones

### Ejercicios «Tu turno»

Están todas en [`soluciones.py`](soluciones.py), con las mismas firmas que en los cuadernos.
`python curso/soluciones.py` las pasa por los correctores.

### Ejercicios del final de cada clase (respuestas breves)

**Clase 0.** (2) ln(512) es la pérdida de un modelo que reparte la probabilidad por igual
entre los 512 tokens: el punto de partida. (3) Un modelo pequeño solo puede aprender un mundo
pequeño; Wikipedia es demasiado variada para un millón de parámetros.

**Clase 1.** (2) Con 260 tokens solo caben los 256 bytes y los 4 especiales: ninguna fusión,
≈1 carácter por token (algo menos por las letras con tilde, que son 2 bytes). (4) El modelo
tiene que aprender por separado los dos tokens: lo que aprenda de uno no pasa automáticamente
al otro. (5) Inglés y código salen troceados en muchos pedazos: el vocabulario se aprendió con cuentos.

**Clase 2.** (1) Más validación = medida más fiable, pero menos datos para aprender. (2) La
validación no se parecería al entrenamiento y la medida engañaría. (3) La validación mediría
memoria, no aprendizaje: saldría mejor de lo que es. (4) 2.048 tokens por paso; unos 2.170
pasos para 4,44 M de tokens. No los ve todos exactamente una vez: las ventanas se sortean.

**Clase 3.** (1) 1.024 × 128 = 131.072. GPT-2: 50.257 × 768 = 38.597.376. (2) Con base 10 las
últimas columnas también giran deprisa: las posiciones lejanas se confunden entre sí.
(3) Con posiciones aprendidas, error: no hay fila 257. Con RoPE, el proyecto también lo
impide (las tablas se calculan hasta `contexto`), pero matemáticamente se podría extender.
(4) Girar ambos el mismo ángulo no cambia el ángulo entre ellos. (5) La posición solo importa
para decidir *a quién* mirar (Q·K); lo que se entrega (V) no necesita girar.

**Clase 4.** (1) Los pesos se van casi enteros a un solo token: el softmax se satura. (2) La
última fila no cambia (ya podía mirarlo todo); la primera es la que más cambia. (3) El primer
token solo puede mirarse a sí mismo. (4) 65.536 siempre: 4 matrices de 128 × 128 sea cual
sea el número de cabezas.

**Clase 5.** (1) 65.536 + 132.096 + 256 = 197.888. (2) 784.896: casi el cuádruple (los
parámetros crecen con el cuadrado del ancho). (4) Normalizar la frase entera mezclaría
información de tokens futuros en los anteriores: rompería la causalidad.

**Clase 6.** (1) ln(4096) = 8,318. (3) (16, 256, 1024) = 4.194.304 números. (4) Cambia el
texto, no la calidad: la semilla controla el azar, no el conocimiento.

**Clase 7.** (1) Medido al preparar el curso (200 pasos, validación final): 0,0003 → 4,99;
0,003 → 2,21; 0,01 → 2,74; 0,03 → 4,10. La zona buena es estrecha y por encima empeora
rápido. No llega a dar `nan` gracias al recorte del gradiente y a AdamW. (2) Los gradientes
se suman a los de las vueltas anteriores y cada actualización es cada vez mayor: el
entrenamiento se degrada. (4) Con 0,05 la validación se queda muy por encima de la del
curso y las muestras son mucho peores. (5) No hay salto: se guardan el optimizador y el
estado del azar (una prueba automática comprueba que pausar y reanudar da el mismo modelo).

**Clase 8.** (2) Suele ser menor: es una frase muy típica del corpus. (3) Sorpresa alta en
casi todos los tokens. (4) Con 300 cuentos tarda más en separarse, pero acaba ocurriendo.

**Clase 9.** (2) Con top-k = 1 no hay azar: la temperatura no cambia cuál es el máximo.
(3) La semilla controla el sorteo, no lo que sabe. (4) Vuelve a su mundo de cuentos.

**Clase 10.** (1) Un cocodrilo no existe en su mundo: suele cambiarlo por un animal conocido.
(2) Aprendería a escribir también las preguntas del usuario; al generar, tras su respuesta
inventaría un turno de usuario. (3) Olvido catastrófico: con una tasa alta, el ajuste borra
parte de lo aprendido en el preentrenamiento y escribe peor.

**Clase 11.** (1) Fallan los nombres con letras o sílabas que casi no aparecen en el corpus
(se trocean en muchos tokens raros). (2) Generaliza bastante bien: en nuestras pruebas
contestó «no lo sé» a «Dime cuál es el planeta más grande», «Explica qué es la gravedad» y
también a «¿Cuánto pesa un gato?», aunque mencione un animal de su mundo. Buscar dónde falla
(peticiones que no empiezan como pregunta, como «Dame un consejo...») es un buen ejercicio. (4) Respuestas con
pasos intermedios (contar de uno en uno, o sumar por cifras): el modelo podría aprender el
procedimiento en vez de la tabla.

**Clase 12.** (3) 20 × 0,92 M ≈ 18 M de tokens: unos 12 minutos en una CPU de 25.000 tokens/s,
unos 2 minutos en una GPU como la del ejemplo.

## 6. Preguntas que salen siempre

**«¿Esto es un LLM de verdad o una simulación?»** Es un LLM de verdad: misma arquitectura,
mismo entrenamiento, mismo ajuste para chat que los grandes. Cambian la escala, los datos y
las etapas posteriores de alineamiento (clase 11, sección 7).

**«¿Por qué no usamos un modelo ya entrenado y lo ajustamos?»** Se puede, y es lo que se hace
en la práctica. Pero el objetivo del curso es entender cada pieza, y eso solo se consigue
viéndolas nacer.

**«¿Entiende lo que dice?»** Lo defendible con lo que medimos está en la clase 11, sección 6.

**«¿Puedo entrenarlo con mis apuntes?»** Sí: `cero nuevo apuntes --receta nano --texto carpeta`.
Con pocos datos memorizará (clase 8).

**«¿Cuánto cuesta entrenar uno grande?»** Órdenes de magnitud: nuestro `nano` ve 3 millones
de tokens en 1-5 minutos de portátil; los modelos comerciales ven billones en miles de GPU
durante meses.

## 7. Si algo falla

| Síntoma | Solución |
|---|---|
| `instalar.cmd` dice que no hay Python | Instalar Python 3.12 de python.org marcando «Add python.exe to PATH» |
| La GPU no se usa | `cero comprobar` lo explica. Suele ser un controlador NVIDIA antiguo: actualizarlo y repetir el instalador. Todo funciona igual en CPU |
| Acentos rotos en la consola | Usar `cero.cmd` (fija UTF-8) en vez de llamar a Python directamente |
| «Ya existe el proyecto» | Otro nombre, o `--reemplazar` para empezarlo de nuevo |
| La pérdida se queda alta o «explota» (`nan`) | Tasa de aprendizaje demasiado alta: bajar `lr` en `config.json` |
| Entrenamiento interrumpido | `cero entrenar --reanudar` (o `cero afinar --reanudar`) |
| La descarga del corpus real se corta | Repetir la orden: continúa donde se quedó. Si llega dañada, se detecta y se descarga otra vez |
| Sin internet en el aula | La receta `nano` y todo el curso funcionan sin conexión. Para `micro`/`mini`/`base`, copiar el archivo `descargas/tinystories_es-00000.parquet` de un equipo a otro |
| GPU sin memoria con la receta `base` | Necesita una GPU de 8 GB libre. Bajar `lote` en `config.json` (y subir `acumular` para compensar) |
| El laboratorio no abre | Mirar la consola de `cero lab`: si el puerto 8765 está ocupado, usa el siguiente libre y lo dice |
