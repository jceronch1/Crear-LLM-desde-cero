// Laboratorio de «LLM desde CERO». JavaScript a secas: sin bibliotecas ni paso de compilación.
"use strict";

// ============================================================================ utilidades
const $ = (selector, raiz = document) => raiz.querySelector(selector);
const $$ = (selector, raiz = document) => [...raiz.querySelectorAll(selector)];
const miles = (n) => Math.round(n).toLocaleString("es-ES");
const decimales = (n, d = 3) => Number(n).toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d });
const duracion = (s) => (s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const visible = (texto) => texto.replace(/\n/g, "↵\n");

/** Crea un elemento: h("p", {class: "nota"}, "texto", otroElemento). */
function h(etiqueta, atributos = {}, ...hijos) {
  const e = document.createElement(etiqueta);
  for (const [clave, valor] of Object.entries(atributos || {})) {
    if (valor === false || valor == null) continue;
    if (clave === "class") e.className = valor;
    else if (clave === "style") e.style.cssText = valor;
    else if (clave.startsWith("on")) e.addEventListener(clave.slice(2), valor);
    else e.setAttribute(clave, valor === true ? "" : valor);
  }
  for (const hijo of hijos.flat(Infinity)) if (hijo != null && hijo !== false) e.append(hijo);
  return e;
}
const vaciar = (e, ...hijos) => { e.replaceChildren(...hijos.flat(Infinity).filter((x) => x != null && x !== false)); return e; };
const cifra = (valor, rotulo) => h("div", { class: "cifra" }, h("b", {}, valor), h("span", {}, rotulo));
const vacio = (texto) => h("div", { class: "vacio" }, texto);

async function api(ruta, datos) {
  const opciones = datos === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(datos) };
  let respuesta;
  try { respuesta = await fetch(ruta, opciones); }
  catch { throw new Error("No se puede hablar con el laboratorio. ¿Sigue abierta la ventana de la consola?"); }
  const cuerpo = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) throw new Error(cuerpo.error || `Error ${respuesta.status}`);
  return cuerpo;
}

/** Lee una respuesta que llega a trozos (una línea JSON por evento). */
async function flujo(ruta, datos, alEvento) {
  const respuesta = await fetch(ruta, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(datos) });
  if (!respuesta.ok) throw new Error((await respuesta.json().catch(() => ({}))).error || `Error ${respuesta.status}`);
  const lector = respuesta.body.getReader();
  const decodificador = new TextDecoder();
  let resto = "";
  for (;;) {
    const { done, value } = await lector.read();
    resto += decodificador.decode(value || new Uint8Array(), { stream: !done });
    const lineas = resto.split("\n");
    resto = lineas.pop();
    for (const linea of lineas) {
      if (!linea.trim()) continue;
      const evento = JSON.parse(linea);
      if (evento.error) throw new Error(evento.error);
      alEvento(evento);
    }
    if (done) break;
  }
}

let temporizadorAviso = null;
function avisar(texto, suave = false) {
  const caja = $("#aviso");
  caja.textContent = texto;
  caja.classList.toggle("suave", suave);
  caja.hidden = false;
  clearTimeout(temporizadorAviso);
  temporizadorAviso = setTimeout(() => { caja.hidden = true; }, 9000);
}
const intentar = (funcion) => async (...args) => { try { return await funcion(...args); } catch (error) { avisar(error.message); } };
function retrasar(funcion, ms = 250) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => funcion(...args), ms); };
}

// ============================================================================ estado global
const estado = { info: null, vista: "inicio", tarea: { activa: false }, etapaGrafica: "base", eventos: [], seguir: true };
const proyecto = () => estado.info && estado.info.proyecto;

async function refrescar() {
  estado.info = await api("/api/estado");
  estado.tarea = estado.info.tarea;
  const selector = $("#selector-proyecto");
  vaciar(selector, estado.info.proyectos.length
    ? estado.info.proyectos.map((n) => h("option", { value: n, selected: n === estado.info.activo }, n))
    : h("option", { value: "" }, "(ninguno todavía)"));
  const p = proyecto();
  const hechas = { datos: p && p.datos, tokens: p && p.datos, modelo: p && p.datos, atencion: p && p.base_listo,
                   entrenar: p && p.base_listo, generar: p && p.base_listo, chat: p && p.chat_listo };
  for (const enlace of $$("#menu a")) enlace.classList.toggle("hecho", Boolean(hechas[enlace.dataset.vista]));
  pintarPie();
}

function pintarPie() {
  const t = estado.tarea;
  const equipo = estado.info ? estado.info.equipo : null;
  vaciar($("#pie-estado"),
    t && t.activa ? h("div", {}, h("b", {}, "● En marcha: "), t.descripcion, ` (${duracion(t.segundos)})`) : null,
    equipo ? h("div", {}, equipo.gpu ? `GPU: ${equipo.gpu}` : `CPU · ${equipo.hilos_pytorch} hilos`) : null,
    estado.info ? h("div", {}, `PyTorch ${estado.info.equipo.pytorch} · cero ${estado.info.version}`) : null);
}

/** Lo que se muestra cuando a una vista le falta una etapa anterior. */
function falta(contenedor, condicion, mensaje) {
  if (condicion) return false;
  vaciar(contenedor, vacio(h("span", {}, mensaje, " ", h("a", { href: "#entrenar" }, "Ir a Entrenar →"))));
  return true;
}

// ============================================================================ navegación
const VISTAS = { inicio: verInicio, datos: verDatos, tokens: verTokens, modelo: verModelo, atencion: verAtencion,
                 entrenar: verEntrenar, generar: verGenerar, chat: verChat };

function navegar() {
  const vista = location.hash.slice(1) in VISTAS ? location.hash.slice(1) : "inicio";
  estado.vista = vista;
  for (const seccion of $$("main > section")) seccion.hidden = seccion.dataset.vista !== vista;
  for (const enlace of $$("#menu a")) enlace.classList.toggle("activa", enlace.dataset.vista === vista);
  $("#aviso").hidden = true;
  intentar(VISTAS[vista])();
  $("#principal").focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

// ============================================================================ inicio
function verInicio() {
  const p = proyecto();
  const pasos = [
    ["datos", "1", "Datos", "El corpus, limpio y repartido", "cero datos", p && p.datos],
    ["tokens", "2", "Tokenizador", "Del texto a los números", "cero tokenizar \"...\"", p && p.datos],
    ["modelo", "3", "Modelo", "El Transformer, pieza a pieza", "cero estado", p && p.datos],
    ["atencion", "4", "Atención", "A qué mira cada token", "", p && p.base_listo],
    ["entrenar", "5", "Entrenar", "De ruido a español", "cero entrenar", p && p.base_listo],
    ["generar", "6", "Generar", "Elegir cada token", "cero generar", p && p.base_listo],
    ["chat", "7", "Conversar", "De LLM a chatbot", "cero afinar · cero chat", p && p.chat_listo],
  ];
  vaciar($("#recorrido"), pasos.map(([vista, n, titulo, descripcion, orden, hecha]) =>
    h("a", { class: "etapa" + (hecha ? " hecha" : ""), href: "#" + vista },
      h("div", { class: "n" }, "PASO " + n), h("div", { class: "t" }, titulo), h("div", { class: "d" }, descripcion),
      orden ? h("div", { class: "o" }, orden) : null)));

  const tarjeta = $("#tarjeta-proyecto");
  if (!p) {
    vaciar(tarjeta, h("h2", {}, "Todavía no hay ningún proyecto"),
      h("p", {}, "Crea el primero con el formulario de la derecha. La receta «nano» no descarga nada y se entrena en unos minutos en cualquier ordenador."),
      h("p", { class: "nota" }, "También puedes hacerlo desde la consola: ", h("code", {}, "cero todo --receta nano")));
  } else {
    const m = p.config.modelo;
    const fila = (rotulo, valor) => [h("dt", {}, rotulo), h("dd", {}, valor)];
    const etapa = (f) => (f ? `${f.estado} · paso ${miles(f.paso)}/${miles(f.pasos)} · validación ${f.perdida_val ?? "—"} · ${duracion(f.segundos)}` : "pendiente");
    vaciar(tarjeta, h("h2", {}, `Proyecto «${p.nombre}»`),
      h("dl", { class: "ficha" },
        fila("receta", `${p.receta} · arquitectura ${p.arquitectura}`),
        fila("modelo", `${m.n_capas} capas · ${m.n_cabezas} cabezas · ancho ${m.d_modelo} · contexto ${m.contexto}`),
        fila("vocabulario", `${miles(m.vocab)} tokens`),
        fila("datos", p.datos ? `${miles(p.datos.partes.entreno.tokens)} tokens · ${p.datos.caracteres_por_token} caracteres/token` : "pendiente"),
        fila("modelo base", etapa(p.base)),
        fila("chatbot", etapa(p.chat)),
        p.base ? fila("parámetros", miles(p.base.parametros)) : null,
        p.base ? fila("entrenado en", p.base.dispositivo.split(" · ")[0]) : null),
      h("div", { class: "fila botones" },
        h("a", { class: "boton", href: "#entrenar" }, "Entrenar"),
        p.base_listo ? h("a", { class: "boton", href: "#generar" }, "Generar") : null,
        p.chat_listo ? h("a", { class: "boton principal", href: "#chat" }, "Conversar") : null));
  }

  const e = estado.info.equipo;
  vaciar($("#tarjeta-equipo"), h("h2", {}, "Este equipo"),
    h("dl", { class: "ficha" },
      [h("dt", {}, "se entrena en"), h("dd", {}, e.gpu ? `${e.gpu}${e.vram_gb ? ` · ${e.vram_gb} GB` : ""} · ${e.precision}` : `CPU · ${e.hilos_pytorch} hilos · ${e.precision}`)],
      [h("dt", {}, "el laboratorio calcula en"), h("dd", {}, estado.info.dispositivo_lab === "cpu" ? "CPU (deja la GPU libre para entrenar)" : "GPU")],
      [h("dt", {}, "programas"), h("dd", {}, `Python ${e.python} · PyTorch ${e.pytorch} · ${e.sistema}`)]),
    h("p", { class: "nota", style: "margin-top:10px" }, "Para saber cuánto tardará cada receta aquí: ", h("code", {}, "cero comprobar")));

  const recetas = estado.info.recetas;
  const selector = $("#nuevo-receta");
  if (!selector.options.length) {
    vaciar(selector, Object.keys(recetas).map((n) => h("option", { value: n }, n)));
    selector.addEventListener("change", notaReceta);
  }
  notaReceta();
}

function notaReceta() {
  const r = estado.info.recetas[$("#nuevo-receta").value];
  $("#nuevo-receta-nota").textContent = `${r.descripcion} ${r.n_capas} capas, ancho ${r.d_modelo}, contexto ${r.contexto}, vocabulario ${miles(r.vocab)}.`
    + (r.corpus === "tinystories" ? " Descarga 258 MB la primera vez." : "");
}

$("#form-nuevo").addEventListener("submit", intentar(async (evento) => {
  evento.preventDefault();
  const f = new FormData(evento.target);
  const nombre = f.get("nombre").trim();
  const existe = estado.info.proyectos.includes(nombre);
  if (existe && !confirm(`Ya existe el proyecto «${nombre}». ¿Empezarlo de nuevo? Se borrará lo que tenía.`)) return;
  estado.tarea = await api("/api/lanzar", { etapa: "todo", nombre, receta: f.get("receta"), arquitectura: f.get("arquitectura"),
                                            dispositivo: f.get("dispositivo"), minutos: f.get("minutos") || null, reemplazar: existe });
  estado.eventos = []; estado.etapaGrafica = "base"; estado.seguir = true;
  location.hash = "#entrenar";
}));

// ============================================================================ datos
async function verDatos() {
  const caja = $("#datos-contenido");
  if (falta(caja, proyecto() && proyecto().datos, "Este proyecto todavía no tiene los datos preparados.")) return;
  const { datos, dialogos, corpus } = await api("/api/datos");
  const partes = datos.partes;
  const ejemplo = datos.ejemplos[0] || "";
  const inicio = ejemplo.split(" ").slice(0, 9).join(" ");
  const piezas = (await api("/api/tokenizar", { texto: inicio })).piezas;
  const celda = (p) => h("td", {}, h("span", { class: "pieza", style: `--tono:${(p.id * 47) % 360}` }, visible(p.texto)));
  const nombres = { sintetico: "cuentos sintéticos (generados por reglas, sin descargas)", tinystories: "TinyStories en español (cuentos infantiles reales)", texto: "tus propios textos" };

  const tarjetas = [
    h("div", { class: "tarjeta" }, h("h2", {}, "El corpus"),
      h("p", {}, "Fuente: ", h("strong", {}, nombres[corpus.fuente] || corpus.fuente), datos.licencia ? ` · licencia ${datos.licencia}` : ""),
      h("div", { class: "cifras" },
        cifra(miles(partes.entreno.documentos), "documentos para entrenar"),
        cifra(miles(partes.entreno.tokens), "tokens para entrenar"),
        cifra(miles(partes.validacion.tokens), "tokens de validación"),
        cifra(decimales(datos.caracteres_por_token, 2), "caracteres por token"),
        cifra(miles(datos.duplicados_descartados), "duplicados descartados")),
      h("p", { class: "nota" }, "La validación es una parte del corpus que el modelo no ve nunca mientras aprende. Sirve para saber si de verdad ha aprendido o solo ha memorizado. El reparto depende del contenido de cada documento: el mismo cuento cae siempre en el mismo lado.")),
    h("div", { class: "tarjeta" }, h("h2", {}, "La tarea: predecir el token siguiente"),
      h("p", { class: "nota" }, "La entrada x y el objetivo y son el mismo texto; y va un token por delante. En cada posición el modelo ve los tokens de x hasta ahí y tiene que acertar el de y."),
      h("div", { class: "tabla-desplazable" }, h("table", {},
        h("tr", {}, h("th", {}, "entrada x"), piezas.slice(0, -1).map(celda)),
        h("tr", {}, h("th", {}, "objetivo y"), piezas.slice(1).map(celda))))),
    h("div", { class: "tarjeta" }, h("h2", {}, "Algunos documentos"),
      datos.ejemplos.slice(0, 3).map((texto) => h("p", { class: "muestra" }, texto.length > 520 ? texto.slice(0, 520) + "…" : texto))),
  ];
  if (dialogos) {
    const total = Object.values(dialogos.categorias).reduce((a, b) => a + b, 0);
    const primero = dialogos.ejemplos.find((e) => e.categoria === "cuento_y_pregunta") || dialogos.ejemplos[0];
    const plantilla = await api("/api/plantilla", { mensajes: primero.mensajes });
    tarjetas.push(h("div", { class: "tarjeta" }, h("h2", {}, "Las conversaciones para el ajuste de chat"),
      h("div", { class: "cifras" }, cifra(miles(dialogos.conversaciones.entreno), "conversaciones"), cifra(dialogos.tokens_por_conversacion, "tokens de media"),
        cifra(miles(dialogos.descartadas_por_largas), "descartadas por largas")),
      h("div", { class: "tabla-desplazable" }, h("table", {}, h("tr", {}, h("th", {}, "tipo"), h("th", { class: "num" }, "cuántas"), h("th", { class: "num" }, "%")),
        Object.entries(dialogos.categorias).map(([nombre, n]) => h("tr", {}, h("td", {}, nombre.replace(/_/g, " ")), h("td", { class: "num" }, miles(n)), h("td", { class: "num" }, decimales(100 * n / total, 1)))))),
      h("h2", { class: "con-margen" }, "Una conversación, tal como la ve el modelo"),
      h("p", { class: "nota" }, "Los tokens oscuros son los especiales. Solo se aprende lo subrayado: lo que dice el asistente y su <|fin|>."),
      h("div", { class: "piezas" }, plantilla.tokens.map((t) => h("span", { class: "pieza" + (t.especial ? " especial" : "") + (t.aprende ? " aprende" : ""), style: `--tono:${(t.id * 47) % 360}` }, visible(t.texto))))));
  }
  vaciar(caja, tarjetas);
}

// ============================================================================ tokenizador
let paginaFusiones = 0;
const POR_PAGINA = 40;

const pintarTokens = intentar(async () => {
  const r = await api("/api/tokenizar", { texto: $("#tok-texto").value || " ", especiales: $("#tok-especiales").checked });
  const conNumeros = $("#tok-numeros").checked;
  vaciar($("#tok-cifras"), cifra(miles(r.caracteres), "caracteres"), cifra(miles(r.bytes), "bytes UTF-8"), cifra(miles(r.tokens), "tokens"),
    cifra(decimales(r.caracteres / Math.max(r.tokens, 1), 2), "caracteres por token"));
  vaciar($("#tok-piezas"), r.piezas.map((p) => h("span", { class: "pieza" + (p.especial ? " especial" : ""), style: `--tono:${(p.id * 47) % 360}`, title: `token ${p.id} · bytes ${p.bytes.join(" ")}` },
    visible(p.texto), conNumeros ? h("small", {}, p.id) : null)));
});

const pintarFusiones = intentar(async () => {
  const r = await api(`/api/fusiones?desde=${paginaFusiones * POR_PAGINA}&cuantas=${POR_PAGINA}`);
  const trozo = (t) => h("span", { class: "pieza", style: "--tono:170" }, visible(t));
  vaciar($("#tok-fusiones"), h("table", {}, h("tr", {}, h("th", { class: "num" }, "nº"), h("th", {}, "se unen"), h("th", {}, ""), h("th", {}, "token nuevo"), h("th", { class: "num" }, "número")),
    r.fusiones.map((f) => h("tr", {}, h("td", { class: "num" }, f.n), h("td", {}, trozo(f.a), " + ", trozo(f.b)), h("td", {}, "→"), h("td", {}, trozo(f.texto)), h("td", { class: "num" }, f.id)))));
  $("#tok-pagina").textContent = `${paginaFusiones * POR_PAGINA + 1}–${Math.min((paginaFusiones + 1) * POR_PAGINA, r.total)} de ${miles(r.total)}`;
  $("#tok-anteriores").disabled = paginaFusiones === 0;
  $("#tok-siguientes").disabled = (paginaFusiones + 1) * POR_PAGINA >= r.total;
  vaciar($("#tok-lista-especiales"), h("table", {}, r.especiales.map((e) => h("tr", {}, h("td", {}, h("span", { class: "pieza especial" }, e.texto)), h("td", { class: "num" }, e.id),
    h("td", {}, { "<|inicio|>": "empieza un documento o una conversación", "<|usuario|>": "a continuación habla la persona", "<|asistente|>": "a continuación habla el modelo", "<|fin|>": "termina un turno: la señal para dejar de generar" }[e.texto] || "")))));
  const partes = [["bytes sueltos", 256, "#8a9099"], ["fusiones aprendidas", r.total, "var(--acento)"], ["especiales", r.especiales.length, "#c98a1b"]];
  vaciar($("#tok-vocab"), h("div", { class: "barra-apilada" }, partes.map(([, n, color]) => h("i", { style: `width:${100 * n / r.vocab}%;background:${color}` }))),
    h("ul", { class: "leyenda" }, partes.map(([nombre, n, color]) => h("li", {}, h("span", { class: "punto", style: `background:${color}` }), nombre, h("span", { class: "v" }, miles(n)))),
      h("li", {}, h("strong", {}, "total"), h("span", { class: "v" }, miles(r.vocab)))));
});

function verTokens() {
  const tarjetas = $$('section[data-vista="tokens"] .tarjeta, section[data-vista="tokens"] .dos-columnas');
  const sin = !(proyecto() && proyecto().datos);
  for (const t of tarjetas) t.hidden = sin;
  let hueco = $("#tok-falta");
  if (!hueco) { hueco = h("div", { id: "tok-falta" }); $('section[data-vista="tokens"]').append(hueco); }
  hueco.hidden = !sin;
  if (sin) { falta(hueco, false, "Este proyecto todavía no tiene tokenizador."); return; }
  pintarTokens();
  pintarFusiones();
}
$("#tok-texto").addEventListener("input", retrasar(pintarTokens, 180));
$("#tok-numeros").addEventListener("change", pintarTokens);
$("#tok-especiales").addEventListener("change", pintarTokens);
$("#tok-anteriores").addEventListener("click", () => { paginaFusiones = Math.max(0, paginaFusiones - 1); pintarFusiones(); });
$("#tok-siguientes").addEventListener("click", () => { paginaFusiones += 1; pintarFusiones(); });

// ============================================================================ modelo
const COLORES = { embeddings: "#8a6fc9", posiciones: "#c96f9b", atencion: "#d99a2b", red_de_avance: "#3b8fd1", normas: "#7c9a3f", salida: "#8a9099" };
const NOMBRES_PIEZA = { embeddings: "embeddings (compartidos con la salida)", posiciones: "posiciones aprendidas", atencion: "atención", red_de_avance: "red de avance", normas: "normas", salida: "capa de salida" };

async function verModelo() {
  const seccion = $('section[data-vista="modelo"]');
  let hueco = $("#modelo-falta");
  if (!hueco) { hueco = h("div", { id: "modelo-falta" }); seccion.append(hueco); }
  const sin = !proyecto();
  for (const t of $$(".tarjeta, .dos-columnas", seccion)) t.hidden = sin;
  hueco.hidden = !sin;
  if (sin) { vaciar(hueco, vacio(h("span", {}, "Todavía no hay ningún proyecto. ", h("a", { href: "#inicio" }, "Crea uno en Inicio →")))); return; }

  const { config: c, reparto, d_cabeza, d_oculta, arquitectura } = await api("/api/modelo");
  const caja = (clase, titulo, detalle) => h("div", { class: "caja " + clase }, h("b", {}, titulo), detalle ? h("small", {}, detalle) : null);
  const flecha = () => h("div", { class: "flecha" }, "↓");
  const norma = () => caja("norma", c.norma === "rms" ? "RMSNorm" : "LayerNorm");
  const atencion = () => caja("atencion", "Atención causal", `${c.n_cabezas} cabezas de ${d_cabeza} dimensiones${c.posicion === "rope" ? " · posición con RoPE" : ""}`);
  const red = () => caja("red", "Red de avance", `${c.mlp === "swiglu" ? "SwiGLU" : c.mlp.toUpperCase()} · ancho interior ${d_oculta}`);
  const suma = () => h("div", { class: "suma" }, "⊕ se suma a lo que entró (atajo residual)");
  const dentro = c.pre_norma
    ? [norma(), flecha(), atencion(), suma(), flecha(), norma(), flecha(), red(), suma()]
    : [atencion(), suma(), flecha(), norma(), flecha(), red(), suma(), flecha(), norma()];
  const posicion = { sinusoidal: "+ tabla sinusoidal de posiciones (fija)", aprendida: "+ tabla de posiciones aprendida", rope: "la posición entra dentro de la atención" }[c.posicion];
  vaciar($("#modelo-diagrama"),
    caja("es", "tokens de entrada", `hasta ${c.contexto} números entre 0 y ${miles(c.vocab - 1)}`), flecha(),
    caja("", "Tabla de embeddings", `${miles(c.vocab)} filas × ${c.d_modelo} · ${posicion}`), flecha(),
    h("div", { class: "bloque-n", "data-veces": `× ${c.n_capas}` }, dentro), flecha(),
    c.pre_norma ? [norma(), flecha()] : null,
    caja("", "Capa de salida", `${c.d_modelo} → ${miles(c.vocab)} puntuaciones${c.atar_pesos ? " · misma matriz que los embeddings" : ""}`), flecha(),
    caja("es", "probabilidad de cada token siguiente", "softmax"));

  const claves = Object.keys(COLORES).filter((k) => reparto[k] > 0);
  vaciar($("#modelo-reparto"), h("div", { class: "cifras" }, cifra(miles(reparto.total), "parámetros")),
    h("div", { class: "barra-apilada" }, claves.map((k) => h("i", { style: `width:${100 * reparto[k] / reparto.total}%;background:${COLORES[k]}`, title: NOMBRES_PIEZA[k] }))),
    h("ul", { class: "leyenda" }, claves.map((k) => h("li", {}, h("span", { class: "punto", style: `background:${COLORES[k]}` }), NOMBRES_PIEZA[k],
      h("span", { class: "v" }, `${miles(reparto[k])} · ${decimales(100 * reparto[k] / reparto.total, 1)} %`)))));

  const filas = [["vocab", "vocabulario", "vocab_size", miles(c.vocab)], ["contexto", "contexto", "context length", c.contexto],
    ["d_modelo", "ancho de cada vector", "d_model", c.d_modelo], ["n_capas", "bloques apilados", "N", c.n_capas],
    ["n_cabezas", "cabezas de atención", "h", c.n_cabezas], ["", "dimensión por cabeza", "d_k", d_cabeza], ["", "ancho de la red de avance", "d_ff", d_oculta]];
  vaciar($("#modelo-config"), h("table", {}, h("tr", {}, h("th", {}, "medida"), h("th", {}, "en el artículo"), h("th", { class: "num" }, "valor")),
    filas.map(([, nombre, articulo, valor]) => h("tr", {}, h("td", {}, nombre), h("td", {}, h("code", {}, articulo)), h("td", { class: "num" }, valor)))));

  const interruptores = [["posicion", "posición"], ["norma", "normalización"], ["pre_norma", "norma antes de la subcapa"], ["mlp", "red de avance"], ["sesgos", "sesgos"], ["dropout", "dropout"]];
  const ver = (v) => (v === true ? "sí" : v === false ? "no" : v === "rms" ? "RMSNorm" : v === "capa" ? "LayerNorm" : String(v));
  const generaciones = estado.info.arquitecturas;
  const rotulos = { paper2017: "paper2017 · «Attention Is All You Need»", gpt2: "gpt2 · 2019", moderna: "moderna · Llama, Qwen, Gemma…" };
  vaciar($("#modelo-generaciones"), h("table", {}, h("tr", {}, h("th", {}, "arquitectura"), interruptores.map(([, n]) => h("th", {}, n))),
    Object.entries(generaciones).map(([nombre, g]) => h("tr", { class: nombre === arquitectura ? "actual" : "" }, h("td", {}, h("strong", {}, rotulos[nombre] || nombre)),
      interruptores.map(([k]) => h("td", {}, ver(g[k])))))));
}

// ============================================================================ atención
const atencion = { datos: null, capa: 0, cabeza: -1, fila: 0 };

const calcularAtencion = intentar(async () => {
  const boton = $("#at-calcular");
  boton.disabled = true;
  try {
    atencion.datos = await api("/api/atencion", { texto: $("#at-texto").value, etapa: $("#at-etapa").value });
    atencion.capa = Math.min(atencion.capa, atencion.datos.atencion.length - 1);
    atencion.cabeza = Math.min(atencion.cabeza, atencion.datos.atencion[0].length - 1);
    atencion.fila = atencion.datos.tokens.length - 1;
    pintarAtencion();
  } finally { boton.disabled = false; }
});

function matrizAtencion() {
  const cabezas = atencion.datos.atencion[atencion.capa];
  if (atencion.cabeza >= 0) return cabezas[atencion.cabeza];
  return cabezas[0].map((fila, i) => fila.map((_, j) => cabezas.reduce((s, c) => s + c[i][j], 0) / cabezas.length));   // media de las cabezas
}

function pintarAtencion() {
  const d = atencion.datos;
  if (!d) return;
  const nCapas = d.atencion.length, nCabezas = d.atencion[0].length;
  const chip = (texto, activa, alPulsar) => h("button", { class: "chip" + (activa ? " activa" : ""), onclick: alPulsar }, texto);
  vaciar($("#at-controles"),
    h("span", { class: "nota" }, "capa"), h("div", { class: "chips" }, [...Array(nCapas)].map((_, i) => chip(i + 1, i === atencion.capa, () => { atencion.capa = i; pintarAtencion(); }))),
    h("span", { class: "nota" }, "cabeza"), h("div", { class: "chips" }, chip("media", atencion.cabeza === -1, () => { atencion.cabeza = -1; pintarAtencion(); }),
      [...Array(nCabezas)].map((_, i) => chip(i + 1, i === atencion.cabeza, () => { atencion.cabeza = i; pintarAtencion(); }))));
  const m = matrizAtencion();
  const rotulo = (t) => visible(t).replace(/\n/g, "");
  vaciar($("#at-mapa"), h("table", { class: "mapa" },
    h("tr", {}, h("th", {}), d.tokens.map((t) => h("th", { class: "col" }, h("span", {}, rotulo(t))))),
    m.map((fila, i) => h("tr", { class: i === atencion.fila ? "elegida" : "" },
      h("th", { class: "fila", onclick: () => { atencion.fila = i; pintarAtencion(); }, title: "Pulsa para ver a qué mira este token" }, rotulo(d.tokens[i])),
      fila.map((v, j) => (j > i ? h("td", { class: "futuro", title: "futuro: no se puede mirar" })
        : h("td", { style: `--p:${Math.min(100, v / 10)}`, title: `«${rotulo(d.tokens[i])}» mira a «${rotulo(d.tokens[j])}»: ${decimales(v / 10, 1)} %` })))))));
  $("#at-nota").textContent = `Modelo «${d.etapa}». Cada fila es un token que pregunta y suma 100 %. La mitad rayada es el futuro: la máscara causal impide mirarlo.`
    + (d.recortado ? ` Solo se muestran los primeros ${d.tokens.length} tokens.` : "");
  const orden = m[atencion.fila].map((v, j) => [v, j]).filter(([, j]) => j <= atencion.fila).sort((a, b) => b[0] - a[0]).slice(0, 8);
  vaciar($("#at-detalle"), h("p", {}, "El token ", h("span", { class: "pieza", style: "--tono:170" }, rotulo(d.tokens[atencion.fila])), ` (posición ${atencion.fila}) reparte así su atención:`),
    h("div", { class: "barras" }, orden.map(([v, j]) => [h("span", { class: "e" }, `${rotulo(d.tokens[j])}  (pos. ${j})`), h("div", {}, h("div", { class: "b", style: `width:${v / 10}%` })), h("span", {}, decimales(v / 10, 1) + " %")])));
}

function verAtencion() {
  const cajas = $$('section[data-vista="atencion"] .tarjeta');
  const sin = !(proyecto() && proyecto().base_listo);
  let hueco = $("#at-falta");
  if (!hueco) { hueco = h("div", { id: "at-falta" }); $('section[data-vista="atencion"]').append(hueco); }
  for (const c of cajas) c.hidden = sin;
  hueco.hidden = !sin;
  if (sin) { falta(hueco, false, "Hace falta un modelo entrenado para mirar su atención."); return; }
  if (!atencion.datos) calcularAtencion();
}
$("#at-calcular").addEventListener("click", calcularAtencion);
$("#at-texto").addEventListener("keydown", (e) => { if (e.key === "Enter") calcularAtencion(); });

// ============================================================================ entrenar
function pintarGrafica() {
  const caja = $("#en-grafica");
  const eventos = estado.eventos;
  const pasosEv = eventos.filter((e) => e.tipo === "paso"), evals = eventos.filter((e) => e.tipo === "eval");
  if (!eventos.length) { vaciar(caja, vacio("Todavía no hay mediciones de esta etapa.")); vaciar($("#en-cifras")); vaciar($("#en-muestras"), vacio("Aparecerán aquí al entrenar.")); $("#en-leyenda").textContent = ""; return; }
  const log = $("#en-log") && $("#en-log").checked;
  const W = 780, H = 300, m = { i: 46, d: 14, a: 12, b: 30 };
  const maxX = Math.max(estado.pasosTotales || 1, ...eventos.map((e) => e.paso));
  const azar = Math.log(estado.vocab || 2);
  const valores = [...pasosEv.map((e) => e.perdida), ...evals.map((e) => e.perdida_val)];
  const maxY = Math.max(azar * 1.04, ...valores), minY = log ? Math.max(0.01, Math.min(...valores) * 0.85) : 0;
  const tf = (v) => (log ? Math.log(Math.max(v, minY)) : v);
  const x = (p) => m.i + (W - m.i - m.d) * p / maxX;
  const y = (v) => m.a + (H - m.a - m.b) * (1 - (tf(v) - tf(minY)) / (tf(maxY) - tf(minY)));
  const camino = (puntos) => puntos.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const marcasY = log ? [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].filter((v) => v >= minY && v <= maxY) : [...Array(Math.floor(maxY) + 1).keys()];
  const marcasX = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(maxX * f));
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Curva de pérdida">
    ${marcasY.map((v) => `<line class="eje" x1="${m.i}" x2="${W - m.d}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.i - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("")}
    ${marcasX.map((v) => `<text x="${x(v)}" y="${H - 8}" text-anchor="middle">${miles(v)}</text>`).join("")}
    <line class="azar" x1="${m.i}" x2="${W - m.d}" y1="${y(azar)}" y2="${y(azar)}"/><text x="${W - m.d - 4}" y="${y(azar) - 5}" text-anchor="end">azar puro: ln(${miles(estado.vocab)}) = ${azar.toFixed(2)}</text>
    <path class="entreno" d="${camino(pasosEv.map((e) => [x(e.paso), y(e.perdida)]))}"/>
    <path class="validacion" d="${camino(evals.map((e) => [x(e.paso), y(e.perdida_val)]))}"/>
    ${evals.map((e) => `<circle class="punto-val" r="3.2" cx="${x(e.paso)}" cy="${y(e.perdida_val)}"><title>paso ${e.paso}: validación ${e.perdida_val}</title></circle>`).join("")}
  </svg>`;
  caja.innerHTML = svg;
  const ultimo = eventos[eventos.length - 1], ultimoPaso = pasosEv[pasosEv.length - 1], mejor = Math.min(...evals.map((e) => e.perdida_val));
  vaciar($("#en-cifras"), cifra(`${miles(ultimo.paso)} / ${miles(maxX)}`, "paso"),
    ultimoPaso ? cifra(decimales(ultimoPaso.perdida), "pérdida de entrenamiento") : null,
    evals.length ? cifra(decimales(mejor), "mejor validación") : null,
    evals.length ? cifra(decimales(Math.exp(mejor), 2), "perplejidad") : null,
    ultimoPaso ? cifra(miles(ultimoPaso.tokens_s), "tokens por segundo") : null,
    cifra(duracion(ultimo.segundos || 0), "tiempo"));
  $("#en-leyenda").replaceChildren("Línea gris: pérdida en los lotes de entrenamiento. Línea de color: pérdida en validación (texto que el modelo no ha visto). ",
    h("label", {}, h("input", { type: "checkbox", id: "en-log", checked: log, onchange: pintarGrafica }), "escala logarítmica"));
  const muestras = evals.filter((e) => e.muestra);
  const cajaMuestras = $("#en-muestras");
  const abajo = cajaMuestras.scrollTop + cajaMuestras.clientHeight >= cajaMuestras.scrollHeight - 30;
  vaciar(cajaMuestras, muestras.length ? muestras.map((e) => h("div", { class: "muestra" }, h("b", {}, `paso ${miles(e.paso)} · validación ${e.perdida_val}`), e.muestra)) : vacio("Aparecerán aquí al entrenar."));
  if (abajo) cajaMuestras.scrollTop = cajaMuestras.scrollHeight;
}

async function cargarMetricas(desdeCero = false) {
  if (!proyecto()) { estado.eventos = []; pintarGrafica(); return; }
  if (desdeCero) estado.eventos = [];
  const r = await api(`/api/metricas?etapa=${estado.etapaGrafica}&desde=${estado.eventos.length}`);
  if (r.total < estado.eventos.length) { estado.eventos = []; pintarGrafica(); return cargarMetricas(); }      // el entrenamiento empezó de nuevo
  estado.pasosTotales = r.pasos; estado.vocab = r.vocab;
  if (r.eventos.length || desdeCero) { estado.eventos.push(...r.eventos); pintarGrafica(); }
}

function pintarTarea() {
  const t = estado.tarea;
  $("#en-parar").disabled = !(t.activa && /Preentrenamiento|Ajuste|Recorrido/.test(t.descripcion || ""));
  for (const b of $$("[data-lanzar]")) b.disabled = Boolean(t.activa);
  if (t.orden) $("#en-registro").textContent = t.registro || "(sin salida todavía)";
  const caja = $("#en-tarea");
  if (!t.orden) { vaciar(caja); return; }
  const ultimo = estado.eventos[estado.eventos.length - 1];
  const fraccion = t.activa && ultimo && estado.pasosTotales ? ultimo.paso / estado.pasosTotales : t.activa ? 0 : 1;
  vaciar(caja, h("div", {}, h("strong", {}, t.activa ? "En marcha: " : t.codigo === 0 ? "Terminado: " : "Terminó con error: "), t.descripcion, " · ", h("code", {}, t.orden), ` · ${duracion(t.segundos)}`),
    t.activa ? h("div", { class: "progreso" }, h("i", { style: `width:${Math.round(100 * fraccion)}%` })) : null);
  if (t.activa) { const reg = $("#en-registro"); reg.scrollTop = reg.scrollHeight; }
}

async function verEntrenar() {
  for (const p of $$("#en-pestanas .pestana")) p.classList.toggle("activa", p.dataset.etapa === estado.etapaGrafica);
  pintarTarea();
  await cargarMetricas(true);
}

for (const boton of $$("[data-lanzar]")) boton.addEventListener("click", intentar(async () => {
  if (!proyecto()) throw new Error("Primero crea un proyecto en Inicio.");
  const etapa = boton.dataset.lanzar;
  estado.tarea = await api("/api/lanzar", { etapa, dispositivo: $("#en-dispositivo").value, minutos: $("#en-minutos").value || null,
                                            pasos: $("#en-pasos").value || null, reanudar: Boolean(boton.dataset.reanudar) });
  if (etapa === "base" || etapa === "chat") { estado.etapaGrafica = etapa; estado.seguir = true; }
  if (!boton.dataset.reanudar) estado.eventos = [];
  await verEntrenar();
}));
$("#en-parar").addEventListener("click", intentar(async () => { await api("/api/parar", {}); avisar("Pausa pedida: el entrenamiento guardará su estado y se detendrá en el próximo paso.", true); }));
for (const pestana of $$("#en-pestanas .pestana")) pestana.addEventListener("click", intentar(async () => { estado.etapaGrafica = pestana.dataset.etapa; estado.seguir = false; await verEntrenar(); }));

/** Cada segundo y medio: ¿hay una tarea en marcha?, ¿hay mediciones nuevas? */
async function latido() {
  try {
    const antes = estado.tarea.activa;
    estado.tarea = await api("/api/tarea");
    const t = estado.tarea;
    if (t.activa && t.proyecto && estado.info && t.proyecto !== estado.info.activo) {                        // «todo» crea y activa un proyecto
      await refrescar();
      estado.eventos = []; atencion.datos = null; charla.mensajes = [];
      if (estado.vista === "entrenar") pintarGrafica();
    }
    if (t.activa && estado.seguir && /Ajuste fino|Ajuste para conversar/.test(t.registro || "") && estado.etapaGrafica !== "chat" && !/Preentrenamiento/.test(t.descripcion)) {
      estado.etapaGrafica = "chat"; estado.eventos = [];
      for (const p of $$("#en-pestanas .pestana")) p.classList.toggle("activa", p.dataset.etapa === "chat");
    }
    if (estado.vista === "entrenar") { if (t.activa || antes) await cargarMetricas(); pintarTarea(); }
    if (antes && !t.activa) {                                // acaba de terminar: todo puede haber cambiado
      await refrescar();
      if (t.codigo !== 0) avisar("La tarea terminó con un error. Mira la consola en la vista Entrenar.");
      if (estado.vista !== "entrenar" && estado.vista !== "generar") intentar(VISTAS[estado.vista])();
    }
    pintarPie();
  } catch { /* el servidor puede estar ocupado un instante; se reintenta en el siguiente latido */ }
}

// ============================================================================ generar
const generacion = { desde: null, ocupado: false };
const mandosGenerar = () => ({ etapa: $("#ge-etapa").value, temperatura: Number($("#ge-temp").value), top_k: Number($("#ge-topk").value) || null,
                               top_p: Number($("#ge-topp").value) < 1 ? Number($("#ge-topp").value) : null });

function pintarMandos() {
  $("#ge-temp-v").textContent = Number($("#ge-temp").value) === 0 ? "0 (siempre el más probable)" : decimales($("#ge-temp").value, 2);
  $("#ge-topk-v").textContent = Number($("#ge-topk").value) === 0 ? "sin límite" : $("#ge-topk").value;
  $("#ge-topp-v").textContent = Number($("#ge-topp").value) >= 1 ? "sin límite" : decimales($("#ge-topp").value, 2);
  $("#chat-temp-v").textContent = decimales($("#chat-temp").value, 2);
}

const pintarCandidatos = intentar(async () => {
  if (!(proyecto() && proyecto().base_listo)) return;
  const r = await api("/api/candidatos", { texto: $("#ge-texto").value, ...mandosGenerar() });
  const maximo = Math.max(...r.candidatos.map((c) => Math.max(c.prob, c.prob_ajustada)), 0.001);
  vaciar($("#ge-candidatos"), r.candidatos.map((c) => [
    h("button", { class: c.prob_ajustada === 0 ? "fuera" : "", title: `token ${c.id}`, onclick: () => { const t = $("#ge-texto"); if (generacion.desde === null) generacion.desde = t.value.length; t.value += c.texto; pintarCandidatos(); } }, c.texto.trim() === "" ? c.texto.replace(/ /g, "␣").replace(/\n/g, "↵") || "∅" : visible(c.texto).replace(/\n/g, "")),
    h("div", { class: "pistas" + (c.prob_ajustada === 0 ? " fuera" : "") }, h("i", { class: "orig", style: `width:${100 * c.prob / maximo}%` }), h("i", { class: "ajus", style: `width:${100 * c.prob_ajustada / maximo}%` })),
    h("span", { class: c.prob_ajustada === 0 ? "fuera" : "" }, `${decimales(100 * c.prob, 1)} % → ${c.prob_ajustada === 0 ? "descartado" : decimales(100 * c.prob_ajustada, 1) + " %"}`)]));
});

async function generarTexto(maximo) {
  if (generacion.ocupado) return;
  const caja = $("#ge-texto");
  generacion.ocupado = true;
  caja.classList.add("generando");
  if (generacion.desde === null) generacion.desde = caja.value.length;
  try {
    await flujo("/api/generar", { texto: caja.value, max: maximo, semilla: $("#ge-semilla").value || null, ...mandosGenerar() }, (e) => {
      if (e.trozo) { caja.value += e.trozo; caja.scrollTop = caja.scrollHeight; }
      if (e.fin) $("#ge-cifras").textContent = `${e.tokens} tokens · ${decimales(e.tokens_s, 0)} por segundo` + (e.se_detuvo ? " · se detuvo solo" : "");
    });
  } finally { generacion.ocupado = false; caja.classList.remove("generando"); pintarCandidatos(); }
}

function verGenerar() {
  const seccion = $('section[data-vista="generar"]');
  const sin = !(proyecto() && proyecto().base_listo);
  let hueco = $("#ge-falta");
  if (!hueco) { hueco = h("div", { id: "ge-falta" }); seccion.append(hueco); }
  for (const t of $$(".tarjeta, .dos-columnas", seccion)) t.hidden = sin;
  hueco.hidden = !sin;
  if (sin) { falta(hueco, false, "Hace falta un modelo entrenado para generar texto."); return; }
  const chat = $('#ge-etapa option[value="chat"]');
  chat.disabled = !proyecto().chat_listo;
  if (chat.disabled) $("#ge-etapa").value = "base";
  pintarMandos();
  pintarCandidatos();
}
$("#ge-generar").addEventListener("click", intentar(() => generarTexto(Number($("#ge-max").value) || 120)));
$("#ge-paso").addEventListener("click", intentar(() => generarTexto(1)));
$("#ge-limpiar").addEventListener("click", () => { if (generacion.desde !== null) $("#ge-texto").value = $("#ge-texto").value.slice(0, generacion.desde); generacion.desde = null; $("#ge-cifras").textContent = ""; pintarCandidatos(); });
$("#ge-texto").addEventListener("input", retrasar(() => { generacion.desde = null; pintarCandidatos(); }, 350));
for (const id of ["ge-temp", "ge-topk", "ge-topp"]) $("#" + id).addEventListener("input", () => { pintarMandos(); retrasarCandidatos(); });
const retrasarCandidatos = retrasar(pintarCandidatos, 200);
$("#ge-etapa").addEventListener("change", pintarCandidatos);
$("#chat-temp").addEventListener("input", pintarMandos);

// ============================================================================ chat
const charla = { mensajes: [], ocupado: false };
const SUGERENCIAS = ["Hola", "¿Quién eres?", "Cuéntame un cuento sobre un gato llamado Milo.", "¿Cómo se llama el gato?", "¿Cuánto es 3 + 4?", "¿Cuál es la capital de Francia?"];

function pintarCharla() {
  const caja = $("#chat-mensajes");
  vaciar(caja, charla.mensajes.length ? charla.mensajes.map((m) => h("div", { class: `burbuja ${m.rol}` + (m.pensando ? " pensando" : "") }, h("span", { class: "quien" }, m.rol === "usuario" ? "tú" : "cero"), m.texto))
    : vacio("Escribe un mensaje o pulsa una de las sugerencias."));
  caja.scrollTop = caja.scrollHeight;
}

async function enviar(texto) {
  texto = texto.trim();
  if (!texto || charla.ocupado) return;
  charla.ocupado = true;
  charla.mensajes.push({ rol: "usuario", texto });
  const respuesta = { rol: "asistente", texto: "", pensando: true };
  const paraElModelo = charla.mensajes.map(({ rol, texto: t }) => ({ rol, texto: t }));
  charla.mensajes.push(respuesta);
  pintarCharla();
  try {
    await flujo("/api/chat", { mensajes: paraElModelo, etapa: $("#chat-etapa").value, temperatura: Number($("#chat-temp").value), top_k: 40 }, (e) => {
      if (e.visto) {
        $("#chat-uso").textContent = `${e.tokens_entrada} de ${e.contexto} tokens de contexto · modelo «${e.etapa}»` + (e.olvidados ? ` · ya no caben los ${e.olvidados} mensajes más antiguos: el modelo no los ve` : "");
        vaciar($("#chat-visto"), e.visto.map((t) => h("span", { class: "pieza" + (t.especial ? " especial" : ""), style: "--tono:170" }, visible(t.texto))));
      }
      if (e.trozo) { respuesta.texto += e.trozo; pintarCharla(); }
    });
    respuesta.texto = respuesta.texto.trim() || "(respuesta vacía)";
  } catch (error) {
    charla.mensajes.splice(-2, 2);
    throw error;
  } finally { respuesta.pensando = false; charla.ocupado = false; pintarCharla(); }
}

function pintarEvaluacion() {
  const p = proyecto();
  const ev = p && p.evaluacion;
  const caja = $("#chat-evaluacion");
  if (!ev) { vaciar(caja, vacio("Todavía no se ha evaluado este proyecto. Pulsa «Evaluar ahora» (tarda alrededor de un minuto).")); return; }
  const partes = [];
  if (ev.base) partes.push(h("div", { class: "cifras" }, cifra(decimales(ev.base.perdida_val), "pérdida de validación (base)"), cifra(decimales(ev.base.perplejidad, 2), "perplejidad"),
    ev.base.bits_por_caracter ? cifra(decimales(ev.base.bits_por_caracter), "bits por carácter") : null));
  if (ev.chat) partes.push(h("div", { class: "tabla-desplazable" }, h("table", {}, h("tr", {}, h("th", {}, "prueba"), h("th", { class: "num" }, "aciertos"), h("th", {}, "un ejemplo")),
    Object.values(ev.chat).map((r) => h("tr", {}, h("td", {}, r.descripcion.trim()), h("td", { class: "num " + (r.aciertos === r.total ? "bien" : r.aciertos === 0 ? "mal" : "") }, `${r.aciertos}/${r.total}`),
      h("td", {}, h("span", { class: r.ejemplo.bien ? "bien" : "mal" }, r.ejemplo.bien ? "✓ " : "✗ "), h("span", { class: "nota" }, r.ejemplo.pregunta.replace(/\n+/g, " ").slice(-90), " → "), r.ejemplo.respuesta.slice(0, 110)))))));
  vaciar(caja, partes);
}

function verChat() {
  const seccion = $('section[data-vista="chat"]');
  const sin = !(proyecto() && proyecto().base_listo);
  let hueco = $("#chat-falta");
  if (!hueco) { hueco = h("div", { id: "chat-falta" }); seccion.append(hueco); }
  for (const t of $$(":scope > .tarjeta, :scope > .dos-columnas", seccion)) t.hidden = sin;
  hueco.hidden = !sin;
  if (sin) { falta(hueco, false, "Hace falta un modelo entrenado para conversar."); return; }
  const opcion = $('#chat-etapa option[value="chat"]');
  opcion.disabled = !proyecto().chat_listo;
  if (opcion.disabled) { $("#chat-etapa").value = "base"; avisar("Este proyecto aún no está ajustado para conversar: el modelo base continuará tu texto como un cuento. Ajústalo en Entrenar.", true); }
  vaciar($("#chat-sugerencias"), SUGERENCIAS.map((s) => h("button", { class: "chip", type: "button", onclick: intentar(() => enviar(s)) }, s)));
  pintarMandos();
  pintarCharla();
  pintarEvaluacion();
}
$("#chat-form").addEventListener("submit", intentar(async (evento) => { evento.preventDefault(); const caja = $("#chat-texto"); const texto = caja.value; caja.value = ""; await enviar(texto); }));
$("#chat-nueva").addEventListener("click", () => { charla.mensajes = []; vaciar($("#chat-visto")); $("#chat-uso").textContent = ""; pintarCharla(); });
$("#chat-evaluar").addEventListener("click", intentar(async () => { estado.tarea = await api("/api/lanzar", { etapa: "evaluar" }); avisar("Evaluando... La tabla se actualizará sola al terminar (alrededor de un minuto).", true); }));

// ============================================================================ arranque
$("#selector-proyecto").addEventListener("change", intentar(async (evento) => {
  if (!evento.target.value) return;
  estado.info = await api("/api/usar", { nombre: evento.target.value });
  atencion.datos = null; charla.mensajes = []; estado.eventos = []; paginaFusiones = 0;
  await refrescar();
  navegar();
}));
window.addEventListener("hashchange", navegar);
(async () => {
  try { await refrescar(); } catch (error) { avisar(error.message); }
  navegar();
  setInterval(latido, 1500);
})();
