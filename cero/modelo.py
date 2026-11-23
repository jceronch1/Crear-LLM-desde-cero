"""El modelo: un Transformer decodificador (tipo GPT), pieza a pieza.

Es la mitad derecha de la Figura 1 de «Attention Is All You Need» (Vaswani et al., 2017),
sin la atención al codificador: para *continuar* un texto no hace falta codificador.

    tokens ─► tabla de embeddings ─► N x Bloque ─► norma ─► capa de salida ─► logits
                (+ posición)          │
                                      ├─ atención causal con varias cabezas   (artículo, §3.2)
                                      └─ red de avance posición a posición    (artículo, §3.3)

Se lee de arriba abajo. Las piezas, en orden:

    NormaRMS            normalizar un vector                       (clase 5 del curso)
    tabla_sinusoidal    posiciones con senos y cosenos, 2017       (clase 3)
    tabla_rope, rotar   posiciones rotando vectores, hoy           (clase 3)
    atencion            la ecuación (1) del artículo, a mano       (clase 4)
    Atencion            lo mismo con varias cabezas y con caché    (clase 4)
    RedDeAvance         la red de dos capas de cada bloque         (clase 5)
    Bloque              atención + red, con atajos residuales      (clase 5)
    GPT                 el modelo completo                         (clase 6)

Los interruptores de `ConfigModelo` permiten montar con este mismo código el decodificador
del artículo de 2017, un GPT-2 o un modelo actual, y compararlos (clase 12).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ConfigModelo


# ======================================================================================
# Normalización
# ======================================================================================
class NormaRMS(nn.Module):
    """RMSNorm: divide cada vector entre su tamaño medio (la raíz de la media de los cuadrados).

    LayerNorm, la del artículo, además resta la media y tiene un sesgo. RMSNorm se salta
    ambas cosas: es más barata y funciona igual de bien, por eso la usan los modelos actuales.

        NormaRMS(x) = x / sqrt(media(x²) + eps) * ganancia
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.ganancia = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()                                   # en precisión mixta, los cuadrados se hacen en 32 bits
        normalizado = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalizado.type_as(x) * self.ganancia


def crear_norma(cfg: ConfigModelo) -> nn.Module:
    if cfg.norma == "rms":
        return NormaRMS(cfg.d_modelo)
    return nn.LayerNorm(cfg.d_modelo, bias=cfg.sesgos)


# ======================================================================================
# Posición: cómo sabe el modelo en qué orden van los tokens
# ======================================================================================
def tabla_sinusoidal(largo: int, d: int) -> Tensor:
    """Codificación posicional del artículo (§3.5). Devuelve una tabla (largo, d).

        PE(pos, 2i)   = sen(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Cada posición recibe un vector fijo que se SUMA al embedding del token. Las primeras
    dimensiones oscilan deprisa (distinguen vecinos) y las últimas despacio (distinguen lejanía).
    """
    posicion = torch.arange(largo, dtype=torch.float32).unsqueeze(1)             # (largo, 1)
    divisor = 10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d)        # (d/2,)
    tabla = torch.zeros(largo, d)
    tabla[:, 0::2] = torch.sin(posicion / divisor)
    tabla[:, 1::2] = torch.cos(posicion / divisor)
    return tabla


def tabla_rope(largo: int, d_cabeza: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Ángulos de RoPE (Su et al., 2021). Devuelve (cos, sin), cada una de forma (largo, d_cabeza/2).

    En vez de sumar un vector de posición, RoPE GIRA las consultas y las claves. El vector de
    cada cabeza se reparte en parejas de números; la pareja i del token en la posición m gira
    un ángulo  m · θ_i,  con  θ_i = base^(-2i/d).  Usa las mismas frecuencias que la tabla
    sinusoidal de 2017: es la misma idea, colocada dentro de la atención.

    La gracia: el producto escalar entre una consulta en m y una clave en n solo depende de
    m - n. La atención ve distancias RELATIVAS y no hay ningún parámetro de posición que aprender.
    """
    frecuencias = 1.0 / (base ** (torch.arange(0, d_cabeza, 2, dtype=torch.float32) / d_cabeza))
    angulos = torch.outer(torch.arange(largo, dtype=torch.float32), frecuencias)
    return angulos.cos(), angulos.sin()


def rotar(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Aplica RoPE: gira cada pareja (x1, x2) de `x` el ángulo que toca a su posición.

    x: (..., T, d_cabeza).  cos, sin: (T, d_cabeza/2).  Es una rotación 2D por pareja:
        x1' = x1·cos - x2·sin        x2' = x1·sin + x2·cos
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ======================================================================================
# Atención
# ======================================================================================
def mascara_causal(n_consultas: int, n_claves: int, device: torch.device | str | None = None) -> Tensor:
    """Matriz booleana (n_consultas, n_claves): True donde SÍ se permite mirar.

    Cada token puede mirarse a sí mismo y a los anteriores, nunca a los que vienen después.
    Normalmente es cuadrada (un triángulo inferior). Con la caché KV hay más claves que
    consultas: las consultas son los tokens nuevos, que ocupan las últimas posiciones.
    """
    todo = torch.ones(n_consultas, n_claves, dtype=torch.bool, device=device)
    return todo.tril(diagonal=n_claves - n_consultas)


def atencion(q: Tensor, k: Tensor, v: Tensor, mascara: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Atención de producto escalar escalado: la ecuación (1) del artículo, línea a línea.

        Atención(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    q: (..., T, d_k) consultas    «qué busco»
    k: (..., S, d_k) claves       «qué ofrezco»
    v: (..., S, d_v) valores      «qué entrego si me eligen»
    Devuelve (salida, pesos); `pesos` es la matriz (..., T, S) de «cuánto mira cada token a cada otro».
    """
    d_k = q.size(-1)
    puntos = q @ k.transpose(-2, -1) / math.sqrt(d_k)       # compatibilidad de cada consulta con cada clave
    if mascara is not None:
        puntos = puntos.masked_fill(~mascara, float("-inf"))  # el futuro queda a -infinito...
    pesos = F.softmax(puntos, dim=-1)                        # ...y el softmax lo convierte en 0 exacto
    return pesos @ v, pesos                                 # promedio de los valores, ponderado por los pesos


class CacheKV:
    """Memoria de claves y valores ya calculados, para generar deprisa.

    Sin caché, escribir el token número t obliga a pasar de nuevo todo el texto anterior por
    todas las capas: n tokens cuestan O(n²). Pero la atención es causal: las claves y los
    valores de los tokens viejos no cambian nunca. Si se guardan, cada token nuevo solo calcula
    lo suyo y mira lo guardado: O(n).
    """

    def __init__(self, n_capas: int) -> None:
        self.k: list[Tensor | None] = [None] * n_capas
        self.v: list[Tensor | None] = [None] * n_capas
        self.longitud = 0                                    # tokens ya guardados

    def agregar(self, capa: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Añade las claves/valores nuevos de una capa y devuelve todos los acumulados."""
        if self.k[capa] is not None:
            k = torch.cat([self.k[capa], k], dim=2)
            v = torch.cat([self.v[capa], v], dim=2)
        self.k[capa], self.v[capa] = k, v
        return k, v


class Atencion(nn.Module):
    """Atención causal con varias cabezas (artículo, §3.2.2).

    Una sola atención solo puede aprender un tipo de relación. Con h cabezas, los d_modelo
    números de cada token se reparten en h grupos de d_cabeza = d_modelo / h, y cada grupo
    hace su propia atención en paralelo. Cuesta lo mismo que una cabeza grande.

        MultiCabeza(x) = Concatenar(cabeza_1, ..., cabeza_h) · W_O
    """

    def __init__(self, cfg: ConfigModelo, capa: int) -> None:
        super().__init__()
        self.capa = capa
        self.n_cabezas, self.d_cabeza = cfg.n_cabezas, cfg.d_cabeza
        d = cfg.d_modelo
        # Las cuatro matrices del artículo: W^Q, W^K, W^V (una fila de cada una por cabeza) y W^O.
        self.W_q = nn.Linear(d, d, bias=cfg.sesgos)
        self.W_k = nn.Linear(d, d, bias=cfg.sesgos)
        self.W_v = nn.Linear(d, d, bias=cfg.sesgos)
        self.W_o = nn.Linear(d, d, bias=cfg.sesgos)

    def forward(
        self,
        x: Tensor,
        rope: tuple[Tensor, Tensor] | None = None,
        cache: CacheKV | None = None,
        ver: list[Tensor] | None = None,
    ) -> Tensor:
        B, T, d = x.shape                                    # lote, tokens, d_modelo
        H, D = self.n_cabezas, self.d_cabeza

        # 1. Proyectar y repartir en cabezas: (B, T, d) -> (B, H, T, D)
        q = self.W_q(x).view(B, T, H, D).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        # 2. RoPE: la posición entra girando consultas y claves (los valores no se tocan).
        if rope is not None:
            q, k = rotar(q, *rope), rotar(k, *rope)

        # 3. Con caché, las claves y valores de los tokens anteriores ya están guardados.
        if cache is not None:
            k, v = cache.agregar(self.capa, k, v)
        S = k.size(2)                                        # claves disponibles (S >= T)

        # 4. La atención en sí.
        if ver is not None:
            # Camino didáctico: la fórmula a la vista, y se guardan los pesos para dibujarlos.
            y, pesos = atencion(q, k, v, mascara_causal(T, S, x.device))
            ver.append(pesos)
        elif S == T:
            # Camino rápido: la misma cuenta dentro de PyTorch, sin guardar la matriz T x T.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v)      # un token nuevo puede mirar todo lo guardado
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mascara_causal(T, S, x.device))

        # 5. Juntar las cabezas y mezclarlas: (B, H, T, D) -> (B, T, d)
        y = y.transpose(1, 2).reshape(B, T, d)
        return self.W_o(y)


# ======================================================================================
# Red de avance
# ======================================================================================
class RedDeAvance(nn.Module):
    """La red «feed-forward» de cada bloque (artículo, §3.3). Trabaja token a token.

    La atención mueve información ENTRE posiciones; esta red la procesa DENTRO de cada una.

        relu / gelu:   sale( activación( entra(x) ) )        con ancho interior 4·d   (2017, GPT-2)
        swiglu:        sale( silu(puerta(x)) * entra(x) )    con ancho interior 8/3·d (modelos actuales)

    En SwiGLU una proyección hace de «puerta»: decide, número a número, cuánto de la otra pasa.
    """

    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        self.tipo = cfg.mlp
        d, oculta = cfg.d_modelo, cfg.d_oculta
        self.entra = nn.Linear(d, oculta, bias=cfg.sesgos)
        self.puerta = nn.Linear(d, oculta, bias=cfg.sesgos) if cfg.mlp == "swiglu" else None
        self.sale = nn.Linear(oculta, d, bias=cfg.sesgos)

    def forward(self, x: Tensor) -> Tensor:
        if self.puerta is not None:
            return self.sale(F.silu(self.puerta(x)) * self.entra(x))
        activacion = F.relu if self.tipo == "relu" else F.gelu
        return self.sale(activacion(self.entra(x)))


# ======================================================================================
# Bloque
# ======================================================================================
class Bloque(nn.Module):
    """Un bloque Transformer: atención y red de avance, cada una con su atajo residual.

    El atajo («x + ...») es lo que permite apilar muchos bloques: cada uno solo tiene que
    aprender una corrección sobre lo que ya había, y el gradiente tiene un camino directo.

        pre-norma  (GPT-2 en adelante):  x = x + Subcapa(Norma(x))
        post-norma (artículo de 2017):   x = Norma(x + Subcapa(x))
    """

    def __init__(self, cfg: ConfigModelo, capa: int) -> None:
        super().__init__()
        self.pre_norma = cfg.pre_norma
        self.norma_1 = crear_norma(cfg)
        self.atencion = Atencion(cfg, capa)
        self.norma_2 = crear_norma(cfg)
        self.red = RedDeAvance(cfg)
        self.soltar = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, rope=None, cache: CacheKV | None = None, ver: list[Tensor] | None = None) -> Tensor:
        if self.pre_norma:
            x = x + self.soltar(self.atencion(self.norma_1(x), rope, cache, ver))
            x = x + self.soltar(self.red(self.norma_2(x)))
        else:
            x = self.norma_1(x + self.soltar(self.atencion(x, rope, cache, ver)))
            x = self.norma_2(x + self.soltar(self.red(x)))
        return x


# ======================================================================================
# El modelo completo
# ======================================================================================
class GPT(nn.Module):
    """Modelo de lenguaje causal: dado un texto, puntúa cuál puede ser el token siguiente."""

    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_modelo

        self.embeddings = nn.Embedding(cfg.vocab, d)         # una fila de d números por cada token del vocabulario
        self.posiciones = nn.Embedding(cfg.contexto, d) if cfg.posicion == "aprendida" else None
        self.soltar = nn.Dropout(cfg.dropout)
        self.bloques = nn.ModuleList([Bloque(cfg, i) for i in range(cfg.n_capas)])
        self.norma_final = crear_norma(cfg) if cfg.pre_norma else nn.Identity()
        self.salida = nn.Linear(d, cfg.vocab, bias=False)    # de d números a una puntuación por token
        if cfg.atar_pesos:
            # «Compartimos la misma matriz de pesos entre los embeddings y la transformación previa al
            # softmax» (artículo, §3.4). Ahorra vocab·d parámetros: mucho, en un modelo pequeño.
            self.salida.weight = self.embeddings.weight

        # Tablas de posición fijas (no son parámetros, no se guardan: se recalculan al cargar).
        if cfg.posicion == "sinusoidal":
            self.register_buffer("tabla_pos", tabla_sinusoidal(cfg.contexto, d), persistent=False)
        elif cfg.posicion == "rope":
            cos, sin = tabla_rope(cfg.contexto, cfg.d_cabeza)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        # El artículo multiplica los embeddings por √d (§3.4) para que la tabla sinusoidal no los tape.
        self.escala = math.sqrt(d) if cfg.posicion == "sinusoidal" else 1.0

        # Inicialización: números pequeños al azar. Las proyecciones que desembocan en el atajo
        # residual empiezan aún más pequeñas (÷ √(2·N)) para que la señal no crezca con la profundidad.
        self.apply(self._iniciar)
        for nombre, p in self.named_parameters():
            if nombre.endswith(("W_o.weight", "sale.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_capas))

    @staticmethod
    def _iniciar(modulo: nn.Module) -> None:
        if isinstance(modulo, (nn.Linear, nn.Embedding)):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        if isinstance(modulo, nn.Linear) and modulo.bias is not None:
            nn.init.zeros_(modulo.bias)

    # ---------------------------------------------------------------------------- hacia delante
    def estados(self, ids: Tensor, cache: CacheKV | None = None, ver: list[Tensor] | None = None) -> Tensor:
        """Todo menos la capa de salida: de (B, T) números de token a (B, T, d) vectores con contexto.

        `cache`: memoria KV para generar token a token.  `ver`: lista donde dejar los pesos de atención.
        """
        B, T = ids.shape
        inicio = cache.longitud if cache is not None else 0  # posición del primer token de esta llamada
        if inicio + T > self.cfg.contexto:
            raise ValueError(
                f"La secuencia llega a {inicio + T} tokens y el modelo solo ve {self.cfg.contexto}. "
                "Recorta el texto (o entrena un modelo con más contexto)."
            )
        x = self.embeddings(ids) * self.escala
        rope = None
        if self.cfg.posicion == "sinusoidal":
            x = x + self.tabla_pos[inicio : inicio + T]
        elif self.cfg.posicion == "aprendida":
            x = x + self.posiciones(torch.arange(inicio, inicio + T, device=ids.device))
        else:
            rope = (self.rope_cos[inicio : inicio + T], self.rope_sin[inicio : inicio + T])
        x = self.soltar(x)
        for bloque in self.bloques:
            x = bloque(x, rope, cache, ver)
        if cache is not None:
            cache.longitud += T
        return self.norma_final(x)

    def forward(self, ids: Tensor, objetivos: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Devuelve (logits, pérdida).

        logits: (B, T, vocab), la puntuación de cada posible token siguiente en cada posición.
        pérdida: entropía cruzada frente a `objetivos` (los mismos tokens desplazados un lugar).
                 Los objetivos que valen -1 no cuentan: así se enmascara lo que no hay que aprender.
        """
        logits = self.salida(self.estados(ids))
        perdida = None
        if objetivos is not None:
            perdida = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), objetivos.reshape(-1), ignore_index=-1)
        return logits, perdida

    @torch.no_grad()
    def mirar(self, ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Para estudiar el modelo por dentro: devuelve (logits, atenciones).

        `atenciones[capa]` tiene forma (B, cabezas, T, T): cuánto mira cada token a cada anterior.
        """
        ver: list[Tensor] = []
        logits = self.salida(self.estados(ids, ver=ver))
        return logits, ver

    def nueva_cache(self) -> CacheKV:
        return CacheKV(self.cfg.n_capas)

    # ---------------------------------------------------------------------------- contabilidad
    def n_parametros(self) -> int:
        """Total de parámetros (los pesos atados se cuentan una sola vez)."""
        return sum(p.numel() for p in self.parameters())

    def reparto(self) -> dict[str, int]:
        """¿Dónde están los parámetros? Un recuento por tipo de pieza."""
        cuenta = {"embeddings": self.embeddings.weight.numel(), "posiciones": 0, "atencion": 0,
                  "red_de_avance": 0, "normas": 0, "salida": 0}
        if self.posiciones is not None:
            cuenta["posiciones"] = self.posiciones.weight.numel()
        if not self.cfg.atar_pesos:
            cuenta["salida"] = self.salida.weight.numel()
        for bloque in self.bloques:
            cuenta["atencion"] += sum(p.numel() for p in bloque.atencion.parameters())
            cuenta["red_de_avance"] += sum(p.numel() for p in bloque.red.parameters())
            cuenta["normas"] += sum(p.numel() for p in bloque.norma_1.parameters())
            cuenta["normas"] += sum(p.numel() for p in bloque.norma_2.parameters())
        cuenta["normas"] += sum(p.numel() for p in self.norma_final.parameters())
        cuenta["total"] = self.n_parametros()
        return cuenta
