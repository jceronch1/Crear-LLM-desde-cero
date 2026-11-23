"""El servidor del laboratorio: una aplicación web local, sin dependencias.

Solo escucha en este mismo ordenador (127.0.0.1) y no envía nada a internet. Sirve la página
(carpeta `web/`) y una pequeña API en JSON con la que la página consulta el modelo:

    GET  /api/estado            proyectos, etapas hechas y datos del equipo
    POST /api/tokenizar         un texto troceado en tokens
    GET  /api/fusiones          las fusiones que aprendió el tokenizador, en orden
    GET  /api/modelo            forma del modelo y reparto de parámetros
    POST /api/atencion          los pesos de atención de cada capa y cabeza para un texto
    POST /api/candidatos        los tokens más probables para continuar un texto
    POST /api/generar           texto generado, entregado a trozos mientras se escribe
    POST /api/chat              respuesta del chatbot, a trozos
    GET  /api/metricas          las mediciones del entrenamiento (para dibujar las curvas)
    POST /api/lanzar            arranca una etapa (datos, entrenar, afinar, evaluar, todo)
    POST /api/parar             pide una pausa ordenada del entrenamiento
    GET  /api/tarea             qué se está ejecutando y sus últimas líneas

Los entrenamientos se lanzan como un proceso aparte (`python -m cero ...`): si algo falla
ahí, el laboratorio sigue vivo, y la memoria de la GPU se libera al terminar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .. import __version__
from .. import dispositivo as disp
from ..chat import codificar_dialogo, preparar_entrada, responder
from ..config import ARQUITECTURAS, RECETAS, ErrorDeConfig, leer_json, receta
from ..entrenar import leer_metricas
from ..generar import candidatos, generar
from ..proyecto import RAIZ, ErrorDeProyecto, Proyecto, activar, activo, listar
from ..tokenizador import INICIO

WEB = Path(__file__).parent / "web"
TIPOS = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
MAX_TEXTO = 4000                 # caracteres como mucho en cualquier texto que llegue de la página
MAX_TOKENS_ATENCION = 48         # una matriz de atención mayor no se lee bien en pantalla


class ErrorDePeticion(ValueError):
    """La página pidió algo que no se puede hacer; el mensaje se le muestra tal cual."""


# ========================================================================================
# Estado del servidor
# ========================================================================================
class Laboratorio:
    def __init__(self, dispositivo: str) -> None:
        self.dispositivo = disp.elegir(dispositivo)
        self.cerrojo = threading.Lock()                      # el modelo atiende una petición cada vez
        self._modelos: dict[tuple[str, str], tuple[float, Any, Any, dict]] = {}
        self.tarea: dict[str, Any] | None = None
        self.proceso: subprocess.Popen | None = None

    # ------------------------------------------------------------------ proyectos y modelos
    def proyecto(self, nombre: str | None = None) -> Proyecto:
        return Proyecto.abrir(nombre or None)

    def modelo(self, proyecto: Proyecto, etapa: str = "auto"):
        """Carga (o reutiliza) un modelo entrenado. Si el archivo cambió en disco, se vuelve a leer."""
        etapa = proyecto.etapa_disponible(etapa)
        ruta = proyecto.mejor(etapa)
        fecha = ruta.stat().st_mtime
        clave = (proyecto.nombre, etapa)
        guardado = self._modelos.get(clave)
        if guardado is None or guardado[0] != fecha:
            modelo, tok, ficha = proyecto.cargar(etapa, self.dispositivo)
            self._modelos = {k: v for k, v in self._modelos.items() if k[0] == proyecto.nombre}   # soltar otros proyectos
            self._modelos[clave] = guardado = (fecha, modelo, tok, ficha)
        return guardado[1], guardado[2], guardado[3], etapa

    # ------------------------------------------------------------------ tareas largas
    def tarea_activa(self) -> bool:
        return self.proceso is not None and self.proceso.poll() is None

    def lanzar(self, orden: list[str], descripcion: str, proyecto: str | None) -> dict[str, Any]:
        if self.tarea_activa():
            raise ErrorDePeticion("Ya hay una tarea en marcha. Espera a que termine o páusala.")
        registro = RAIZ / "proyectos" if not os.environ.get("CERO_PROYECTOS") else Path(os.environ["CERO_PROYECTOS"])
        registro.mkdir(parents=True, exist_ok=True)
        ruta = registro / ".laboratorio.log"
        entorno = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        archivo = open(ruta, "w", encoding="utf-8")
        self.proceso = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "cero", *orden], cwd=RAIZ, env=entorno,
                                        stdout=archivo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        archivo.close()
        self.tarea = {"orden": "cero " + " ".join(orden), "descripcion": descripcion, "proyecto": proyecto,
                      "inicio": time.time(), "registro": str(ruta)}
        return self.estado_tarea()

    def estado_tarea(self) -> dict[str, Any]:
        if self.tarea is None:
            return {"activa": False}
        codigo = self.proceso.poll() if self.proceso else 0
        texto = ""
        try:
            texto = Path(self.tarea["registro"]).read_text(encoding="utf-8", errors="replace")[-6000:]
        except OSError:
            pass
        return {"activa": codigo is None, "codigo": codigo, "orden": self.tarea["orden"],
                "descripcion": self.tarea["descripcion"], "proyecto": self.tarea["proyecto"],
                "segundos": round(time.time() - self.tarea["inicio"], 1), "registro": texto}


# ========================================================================================
# Lo que responde cada dirección
# ========================================================================================
def api_estado(lab: Laboratorio, _datos: dict) -> dict:
    nombres = listar()
    actual = activo()
    proyecto = None
    if actual:
        proyecto = Proyecto(actual).estado()
        ruta = Proyecto(actual).carpeta / "evaluacion.json"
        proyecto["evaluacion"] = leer_json(ruta) if ruta.exists() else None
    return {
        "version": __version__,
        "proyectos": nombres,
        "activo": actual,
        "proyecto": proyecto,
        "equipo": disp.describir(disp.elegir("auto")),
        "dispositivo_lab": lab.dispositivo,
        "recetas": {n: {"descripcion": r["descripcion"], "corpus": r["corpus"]["fuente"], "vocab": r["vocab"],
                        **{k: r["modelo"][k] for k in ("contexto", "d_modelo", "n_capas", "n_cabezas")},
                        "pasos": r["entreno"]["pasos"], "pasos_chat": r["chat"]["pasos"]} for n, r in RECETAS.items()},
        "arquitecturas": ARQUITECTURAS,
        "tarea": lab.estado_tarea(),
    }


def api_usar(lab: Laboratorio, datos: dict) -> dict:
    proyecto = Proyecto.abrir(_texto(datos, "nombre", 64))
    activar(proyecto.nombre)
    return api_estado(lab, {})


def api_tokenizar(lab: Laboratorio, datos: dict) -> dict:
    tok = lab.proyecto(datos.get("proyecto")).tokenizador()
    texto = _texto(datos, "texto")
    piezas = tok.piezas(texto, especiales=bool(datos.get("especiales", True)))
    return {"piezas": piezas, "caracteres": len(texto), "bytes": len(texto.encode("utf-8")), "tokens": len(piezas),
            "vocab": tok.vocab}


def api_fusiones(lab: Laboratorio, datos: dict) -> dict:
    tok = lab.proyecto(datos.get("proyecto")).tokenizador()
    desde = max(0, int(datos.get("desde", 0)))
    cuantas = max(1, min(200, int(datos.get("cuantas", 60))))
    lista = [{"n": i + 1, "id": 256 + i, "a": tok.texto_de(a), "b": tok.texto_de(b), "texto": tok.texto_de(256 + i)}
             for i, (a, b) in enumerate(tok.fusiones[desde : desde + cuantas], start=desde)]
    return {"fusiones": lista, "total": len(tok.fusiones), "vocab": tok.vocab,
            "especiales": [{"id": i, "texto": t} for t, i in tok.especiales.items()]}


def api_modelo(lab: Laboratorio, datos: dict) -> dict:
    from ..modelo import GPT

    proyecto = lab.proyecto(datos.get("proyecto"))
    cfg = proyecto.config_modelo()
    modelo = GPT(cfg)                                        # sin pesos entrenados: solo para contar
    return {"config": cfg.a_dict(), "d_cabeza": cfg.d_cabeza, "d_oculta": cfg.d_oculta, "reparto": modelo.reparto(),
            "arquitectura": proyecto.config.get("arquitectura"), "receta": proyecto.config.get("receta")}


def api_datos(lab: Laboratorio, datos: dict) -> dict:
    estado = lab.proyecto(datos.get("proyecto")).estado()
    return {"datos": estado["datos"], "dialogos": estado["dialogos"], "corpus": estado["config"]["corpus"]}


def api_atencion(lab: Laboratorio, datos: dict) -> dict:
    proyecto = lab.proyecto(datos.get("proyecto"))
    texto = _texto(datos, "texto")
    with lab.cerrojo:
        modelo, tok, _, etapa = lab.modelo(proyecto, datos.get("etapa") or "auto")
        ids = ([tok.id(INICIO)] + tok.codificar_texto(texto))[:MAX_TOKENS_ATENCION]
        x = torch.tensor([ids], device=lab.dispositivo)
        _, atenciones = modelo.mirar(x)
    return {"etapa": etapa, "tokens": [tok.texto_de(i) for i in ids],
            "recortado": len(tok.codificar_texto(texto)) + 1 > MAX_TOKENS_ATENCION,
            # atencion[capa][cabeza][fila que pregunta][columna a la que mira], en tantos por mil
            "atencion": [[[[round(float(v) * 1000) for v in fila] for fila in cabeza] for cabeza in capa[0].cpu()]
                         for capa in atenciones]}


def _mandos(datos: dict) -> dict:
    temperatura = float(datos.get("temperatura", 0.8))
    top_k = datos.get("top_k")
    top_p = datos.get("top_p")
    return {"temperatura": max(0.0, min(temperatura, 3.0)),
            "top_k": int(top_k) if top_k else None,
            "top_p": float(top_p) if top_p and 0 < float(top_p) < 1 else None}


def api_candidatos(lab: Laboratorio, datos: dict) -> dict:
    proyecto = lab.proyecto(datos.get("proyecto"))
    with lab.cerrojo:
        modelo, tok, _, etapa = lab.modelo(proyecto, datos.get("etapa") or "auto")
        if datos.get("mensajes"):
            ids, _ = preparar_entrada(tok, _mensajes(datos), modelo.cfg.contexto, 8)
        else:
            ids = [tok.id(INICIO)] + tok.codificar_texto(_texto(datos, "texto", vacio=True))
        lista = candidatos(modelo, tok, ids, cuantos=14, **_mandos(datos))
    return {"etapa": etapa, "candidatos": lista, "tokens_entrada": len(ids), "contexto": modelo.cfg.contexto}


def flujo_generar(lab: Laboratorio, datos: dict):
    proyecto = lab.proyecto(datos.get("proyecto"))
    texto = _texto(datos, "texto", vacio=True)
    maximo = max(1, min(600, int(datos.get("max", 120))))
    semilla = datos.get("semilla")
    with lab.cerrojo:
        modelo, tok, _, etapa = lab.modelo(proyecto, datos.get("etapa") or "auto")
        ids = [tok.id(INICIO)] + tok.codificar_texto(texto)
        yield {"etapa": etapa}
        decodificador, cuenta, inicio = tok.flujo(), 0, time.perf_counter()
        for token in generar(modelo, ids, maximo, parar=tok.especiales.values(),
                             semilla=int(semilla) if semilla not in (None, "") else None, **_mandos(datos)):
            cuenta += 1
            trozo = decodificador.agregar(token)
            if trozo:
                yield {"trozo": trozo}
        resto = decodificador.terminar()
        if resto:
            yield {"trozo": resto}
        segundos = time.perf_counter() - inicio
        yield {"fin": True, "tokens": cuenta, "tokens_s": round(cuenta / max(segundos, 1e-9), 1), "se_detuvo": cuenta < maximo}


def flujo_chat(lab: Laboratorio, datos: dict):
    proyecto = lab.proyecto(datos.get("proyecto"))
    mensajes = _mensajes(datos)
    maximo = max(1, min(600, int(datos.get("max", 220))))
    semilla = datos.get("semilla")
    with lab.cerrojo:
        modelo, tok, _, etapa = lab.modelo(proyecto, datos.get("etapa") or "auto")
        contexto = modelo.cfg.contexto
        ids, olvidados = preparar_entrada(tok, mensajes, contexto, min(maximo, contexto // 3))
        yield {"etapa": etapa, "olvidados": olvidados, "tokens_entrada": len(ids), "contexto": contexto,
               "visto": [{"texto": tok.texto_de(i), "especial": i in tok.especiales.values()} for i in ids]}
        cuenta, inicio = 0, time.perf_counter()
        for trozo in responder(modelo, tok, mensajes, maximo, semilla=int(semilla) if semilla not in (None, "") else None,
                               **_mandos(datos)):
            cuenta += 1
            yield {"trozo": trozo}
        yield {"fin": True, "segundos": round(time.perf_counter() - inicio, 2)}


def api_plantilla(lab: Laboratorio, datos: dict) -> dict:
    tok = lab.proyecto(datos.get("proyecto")).tokenizador()
    ids, mascara = codificar_dialogo(tok, _mensajes(datos), abrir_respuesta=bool(datos.get("abrir")))
    return {"tokens": [{"texto": tok.texto_de(i), "id": i, "especial": i in tok.especiales.values(), "aprende": bool(m)}
                       for i, m in zip(ids, mascara)]}


def api_metricas(lab: Laboratorio, datos: dict) -> dict:
    proyecto = lab.proyecto(datos.get("proyecto"))
    etapa = datos.get("etapa") if datos.get("etapa") in ("base", "chat") else "base"
    eventos = leer_metricas(proyecto.etapa(etapa))
    desde = max(0, int(datos.get("desde", 0)))
    return {"etapa": etapa, "eventos": eventos[desde:], "total": len(eventos),
            "pasos": proyecto.config["entreno" if etapa == "base" else "chat"]["pasos"],
            "vocab": proyecto.config["modelo"]["vocab"]}


def api_lanzar(lab: Laboratorio, datos: dict) -> dict:
    etapa = datos.get("etapa")
    dispositivo = datos.get("dispositivo") if datos.get("dispositivo") in ("auto", "cpu", "gpu") else "auto"
    extra: list[str] = []
    if datos.get("minutos"):
        extra += ["--minutos", str(max(0.2, min(600.0, float(datos["minutos"]))))]
    elif datos.get("pasos"):
        extra += ["--pasos", str(max(10, min(1_000_000, int(datos["pasos"]))))]
    if etapa == "todo":
        nombre = _texto(datos, "nombre", 64)
        nombre_receta = datos.get("receta", "nano")
        receta(nombre_receta, datos.get("arquitectura", "moderna"))          # valida antes de lanzar
        Proyecto(nombre)                                                    # valida el nombre
        orden = ["todo", nombre, "--receta", nombre_receta, "--arquitectura", datos.get("arquitectura", "moderna"),
                 "--dispositivo", dispositivo] + (["--reemplazar"] if datos.get("reemplazar") else [])
        if datos.get("minutos"):
            orden += extra
        return lab.lanzar(orden, f"Recorrido completo con la receta «{nombre_receta}»", nombre)
    proyecto = lab.proyecto(datos.get("proyecto"))
    con_proyecto = ["-p", proyecto.nombre]
    if etapa == "datos":
        return lab.lanzar(["datos", *con_proyecto], "Datos y tokenizador", proyecto.nombre)
    if etapa in ("base", "chat"):
        orden = ["entrenar" if etapa == "base" else "afinar", *con_proyecto, "--dispositivo", dispositivo]
        orden += ["--reanudar"] if datos.get("reanudar") else extra
        return lab.lanzar(orden, "Preentrenamiento" if etapa == "base" else "Ajuste para conversar", proyecto.nombre)
    if etapa == "evaluar":
        return lab.lanzar(["evaluar", *con_proyecto, "--dispositivo", dispositivo], "Evaluación", proyecto.nombre)
    raise ErrorDePeticion(f"Etapa desconocida: {etapa!r}")


def api_parar(lab: Laboratorio, _datos: dict) -> dict:
    if not lab.tarea_activa() or not lab.tarea or not lab.tarea.get("proyecto"):
        raise ErrorDePeticion("No hay ningún entrenamiento en marcha.")
    # El entrenamiento mira si existe este archivo después de cada paso; al verlo, guarda y se detiene.
    (Proyecto(lab.tarea["proyecto"]).carpeta / "PARAR").write_text("pausa pedida desde el laboratorio", encoding="utf-8")
    return lab.estado_tarea()


def api_tarea(lab: Laboratorio, _datos: dict) -> dict:
    return lab.estado_tarea()


RUTAS = {"/api/estado": api_estado, "/api/usar": api_usar, "/api/tokenizar": api_tokenizar, "/api/fusiones": api_fusiones,
         "/api/modelo": api_modelo, "/api/datos": api_datos, "/api/atencion": api_atencion, "/api/candidatos": api_candidatos,
         "/api/plantilla": api_plantilla, "/api/metricas": api_metricas, "/api/lanzar": api_lanzar, "/api/parar": api_parar,
         "/api/tarea": api_tarea}
FLUJOS = {"/api/generar": flujo_generar, "/api/chat": flujo_chat}
SOLO_POST = {"/api/usar", "/api/lanzar", "/api/parar"}


def _texto(datos: dict, clave: str, maximo: int = MAX_TEXTO, vacio: bool = False) -> str:
    valor = datos.get(clave, "")
    if not isinstance(valor, str) or (not valor and not vacio):
        raise ErrorDePeticion(f"Falta «{clave}».")
    if len(valor) > maximo:
        raise ErrorDePeticion(f"«{clave}» es demasiado largo (máximo {maximo} caracteres).")
    return valor


def _mensajes(datos: dict) -> list[dict[str, str]]:
    mensajes = datos.get("mensajes")
    if not isinstance(mensajes, list) or not mensajes or len(mensajes) > 60:
        raise ErrorDePeticion("Falta la conversación.")
    limpios = []
    for m in mensajes:
        if not isinstance(m, dict) or m.get("rol") not in ("usuario", "asistente") or not isinstance(m.get("texto"), str):
            raise ErrorDePeticion("Conversación con formato incorrecto.")
        limpios.append({"rol": m["rol"], "texto": m["texto"][:MAX_TEXTO]})
    return limpios


# ========================================================================================
# El servidor HTTP
# ========================================================================================
def crear_manejador(lab: Laboratorio, puerto: int):
    permitidos = {f"127.0.0.1:{puerto}", f"localhost:{puerto}"}

    class Manejador(BaseHTTPRequestHandler):
        server_version = "cero-lab"

        def log_message(self, *_args: Any) -> None:          # sin una línea en consola por cada petición
            pass

        # -------------------------------------------------------------- respuestas
        def _json(self, codigo: int, valor: Any) -> None:
            cuerpo = json.dumps(valor, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _linea(self, evento: dict) -> None:
            """Envía un evento de una respuesta por trozos: una línea de JSON, y se vacía el búfer."""
            self.wfile.write((json.dumps(evento, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        def _local(self) -> bool:
            """Solo se atiende a esta misma máquina y a esta misma página (no a otras webs abiertas)."""
            if self.headers.get("Host", "") not in permitidos:
                return False
            origen = self.headers.get("Origin")
            return origen is None or origen in {f"http://{h}" for h in permitidos}

        def _atender(self, ruta: str, datos: dict) -> None:
            try:
                if ruta in FLUJOS:
                    eventos = FLUJOS[ruta](lab, datos)
                    try:
                        primero = next(eventos)              # aquí se valida la petición: si falla, sale un error 400 normal
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Accel-Buffering", "no")
                        self.end_headers()                   # sin Content-Length: el final lo marca el cierre
                        try:
                            self._linea(primero)
                            for evento in eventos:
                                self._linea(evento)
                        except (ErrorDeProyecto, ErrorDeConfig, ErrorDePeticion, ValueError) as error:
                            self._linea({"error": str(error)})
                    finally:
                        eventos.close()                      # suelta el modelo aunque la página se cierre a mitad
                    return
                self._json(200, RUTAS[ruta](lab, datos))
            except (ErrorDeProyecto, ErrorDeConfig, ErrorDePeticion) as error:
                self._json(400, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self._json(400, {"error": f"Petición incorrecta: {error}"})
            except (BrokenPipeError, ConnectionError):
                pass                                         # la página se cerró a mitad de una respuesta
            except Exception as error:  # noqa: BLE001  (cualquier otro fallo: que la página lo pueda mostrar)
                self._json(500, {"error": f"Error interno: {type(error).__name__}: {error}"})

        # -------------------------------------------------------------- verbos
        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if not self._local():
                return self._json(403, {"error": "Solo se atiende desde este mismo equipo."})
            if url.path in RUTAS and url.path not in SOLO_POST:
                return self._atender(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
            nombre = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
            archivo = (WEB / nombre).resolve()
            if WEB.resolve() not in archivo.parents or not archivo.is_file() or archivo.suffix not in TIPOS:
                return self._json(404, {"error": "No existe."})
            cuerpo = archivo.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TIPOS[archivo.suffix])
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if not self._local():
                return self._json(403, {"error": "Solo se atiende desde este mismo equipo."})
            if url.path not in RUTAS and url.path not in FLUJOS:
                return self._json(404, {"error": "No existe."})
            # Exigir JSON impide que otra página web abierta en el navegador dispare órdenes a escondidas.
            if "application/json" not in self.headers.get("Content-Type", ""):
                return self._json(415, {"error": "Se esperaba JSON."})
            try:
                largo = int(self.headers.get("Content-Length", "0"))
                if largo > 2_000_000:
                    return self._json(413, {"error": "Petición demasiado grande."})
                datos = json.loads(self.rfile.read(largo) or b"{}")
                if not isinstance(datos, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "JSON incorrecto."})
            self._atender(url.path, datos)

    return Manejador


def servir(puerto: int = 8765, abrir: bool = True, dispositivo: str = "cpu") -> None:
    """Arranca el laboratorio y se queda atendiendo hasta que se pulse Ctrl+C."""
    lab = Laboratorio(dispositivo)
    servidor = None
    for intento in range(20):                                # si el puerto está ocupado, se prueba el siguiente
        try:
            servidor = ThreadingHTTPServer(("127.0.0.1", puerto + intento), crear_manejador(lab, puerto + intento))
            puerto += intento
            break
        except OSError:
            continue
    if servidor is None:
        raise ErrorDeProyecto(f"No hay ningún puerto libre entre {puerto} y {puerto + 19}.")
    servidor.daemon_threads = True
    direccion = f"http://127.0.0.1:{puerto}"
    print(f"\n  Laboratorio de «LLM desde CERO» en  {direccion}")
    print(f"  Los modelos se ejecutan en: {disp.resumen(lab.dispositivo)}")
    print("  Deja esta ventana abierta mientras lo usas. Ctrl+C para cerrarlo.\n", flush=True)
    if abrir:
        threading.Timer(0.6, webbrowser.open, args=(direccion,)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n  Laboratorio cerrado.")
    finally:
        servidor.server_close()
        if lab.tarea_activa():
            print("  (aviso) Sigue en marcha la tarea que lanzaste; terminará sola.")
