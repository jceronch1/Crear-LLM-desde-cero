# Referencia de la orden `cero`

[← Volver al README](../README.md)

Desde la carpeta del proyecto: `.\cero <orden>` en Windows y `./cero.sh <orden>` en Linux y
macOS (o `cero <orden>` con el entorno activado). Cada orden explica sus opciones con `--help`,
por ejemplo `.\cero entrenar --help`.

## El recorrido

| Orden | Qué hace |
|---|---|
| `cero comprobar [--rapido]` | Comprueba la instalación y mide cuánto tardará cada receta en este equipo |
| `cero todo [nombre] --receta R [--arquitectura A] [--minutos N]` | Todo seguido: datos, preentrenamiento y ajuste para chat |
| `cero nuevo nombre --receta R [--arquitectura A] [--texto ruta]` | Crea un proyecto sin entrenarlo |
| `cero datos` | Etapa 1: lee el corpus, entrena el tokenizador y lo convierte todo en tokens |
| `cero entrenar [--pasos N \| --minutos N] [--reanudar]` | Etapa 2: preentrena el modelo base. Ctrl+C pausa y guarda |
| `cero afinar [--pasos N \| --minutos N] [--reanudar]` | Etapa 3: lo ajusta para conversar |

## Usar el modelo

| Orden | Qué hace |
|---|---|
| `cero chat` | Conversación en la consola. Dentro: `/nuevo` borra la conversación, `/dentro` muestra los tokens que ve el modelo, `/salir` termina |
| `cero generar "texto" [--temperatura T] [--top-k K] [--top-p P] [--semilla S] [--muestras N]` | Continúa un texto |
| `cero evaluar [--casos N]` | Pérdida, perplejidad y bits por carácter del modelo base; aciertos del chatbot en preguntas nuevas |
| `cero tokenizar "texto"` | Cómo trocea un texto el tokenizador del proyecto |
| `cero lab [--dispositivo gpu] [--puerto P]` | El laboratorio en el navegador (calcula en CPU por defecto, para dejar la GPU libre al entrenar) |
| `cero curso` | Abre las clases en JupyterLab |

## Proyectos

| Orden | Qué hace |
|---|---|
| `cero recetas` | Lista las recetas y las arquitecturas |
| `cero proyectos` | Lista los proyectos; el marcado con ► es el activo |
| `cero usar nombre` | Cambia el proyecto activo |
| `cero estado` | Qué etapas del proyecto están hechas |

Las órdenes trabajan sobre el proyecto activo; `-p nombre` elige otro. Las que entrenan o usan el
modelo aceptan `--dispositivo auto|cpu|gpu` (por defecto, la GPU si la hay).

## Opciones de las recetas

| Opción | Valores |
|---|---|
| `--receta` | `nano` (0,9 M) · `micro` (5,8 M) · `mini` (15,7 M) · `base` (35,7 M) |
| `--arquitectura` | `moderna` (RoPE, RMSNorm, SwiGLU) · `gpt2` · `paper2017` (el artículo original) |
| `--texto` | Un archivo `.txt` o una carpeta con varios: entrenar con tus propios textos |
| `--minutos N` | Ajusta los pasos para que el entrenamiento dure unos N minutos en este equipo |
| `--reemplazar` | Si el proyecto ya existe, empezarlo de nuevo |

## Dentro de un proyecto

```text
proyectos/<nombre>/
├── config.json          la receta: se puede editar antes de cada etapa
├── tokenizador.json     el vocabulario aprendido
├── datos/               el corpus en tokens y las conversaciones para el ajuste
├── base/                el modelo que continúa textos
│   ├── mejor.pt         el mejor según la validación (lleva dentro el tokenizador)
│   ├── ultimo.pt        todo lo necesario para reanudar
│   └── metricas.jsonl   las mediciones del entrenamiento
├── chat/                el chatbot (misma estructura)
└── evaluacion.json      el resultado de `cero evaluar`
```

En `config.json` se pueden cambiar, por ejemplo, `modelo.n_capas`, `modelo.d_modelo`,
`entreno.pasos`, `entreno.lr` o `chat.pasos`. Qué hay que repetir después:

| Si cambias... | Repite... |
|---|---|
| `vocab` o el corpus | `cero datos`, `cero entrenar` y `cero afinar` |
| `modelo` o `entreno` | `cero entrenar` y `cero afinar` |
| `chat` | `cero afinar` |

## Variables de entorno

| Variable | Efecto |
|---|---|
| `CERO_DISPOSITIVO=cpu` | Lo que elegiría la GPU usa la CPU (por ejemplo, para todo un aula) |
| `CERO_PROYECTOS=ruta` | Guardar los proyectos en otra carpeta |
| `CERO_DESCARGAS=ruta` | Guardar el corpus descargado en otra carpeta |
