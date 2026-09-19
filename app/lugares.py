"""
lugares.py - Lugares y fechas citables, filtro geografico-temporal y puntos del mapa.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Traido del notebook de la Etapa 2 (retrieval v2 sobre la metadata enriquecida) y
adaptado a la arquitectura del equipo: codigo puro, sin LLM y sin dependencias
pesadas, para que se pueda probar sin la base vectorial.

Cada fragmento de la metadata trae:

    geo    : [{nombre, canonico, iso, clase, lat, lon, verificado, origen}]
    fechas : [{valor, origen}]      origen = "directo" | "documento"

Reglas de uso, decididas y auditadas durante la preparacion del corpus:

  FILTRAR  por `iso` o `canonico`, nunca por `nombre`: el nombre es la forma
           literal del texto y tiene variantes (Brasil/Brazil, Mexico/México).
  MAPEAR   solo entidades con `verificado=True` y coordenada.
  CITAR    con `nombre`, la forma literal del texto (100 % de literalidad medida).

Tres usos en el sistema:

  1. Atributos citables. El redactor y el comparador reciben, junto a cada
     fragmento, los lugares y las fechas de EVENTO que aparecen escritos en el.
     El prompt solo les permite atribuir un lugar o una fecha a una cita si esta
     en esa lista. Es la defensa contra la alucinacion geografica mas comun: un
     lugar correcto pegado a un documento que nunca lo menciona.
  2. Preferencia geografico-temporal. Si la pregunta nombra un pais o un anio,
     buscar_corpus antepone los fragmentos que los contienen. Es una preferencia,
     no un filtro duro: si no alcanzan, se completa con el ranking normal.
  3. Puntos del mapa de una respuesta de texto: los lugares que la respuesta
     MENCIONA, tomados solo de los fragmentos que CITA. Asi el mapa ilustra la
     respuesta y no todo el corpus.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable, Optional

from .seguridad import normalizar

MAX_LUGARES = 8
MAX_FECHAS = 6
ANIO_MIN, ANIO_MAX = 1900, 2035


def _lista(valor) -> list:
    """El campo como lista. pandas entrega NaN cuando una fila no trae el campo."""
    return valor if isinstance(valor, list) else []


def _verificadas(geo) -> list[dict]:
    return [g for g in _lista(geo) if isinstance(g, dict) and g.get("verificado")]


# ---------------------------------------------------------------------------
# 1. Atributos citables de un fragmento
# ---------------------------------------------------------------------------
def lugares_citables(geo) -> list[str]:
    """Lugares escritos en el fragmento, en su forma literal (`nombre`)."""
    salida: list[str] = []
    for g in _verificadas(geo):
        nombre = str(g.get("nombre") or "").strip()
        if nombre and nombre not in salida:
            salida.append(nombre)
    return salida[:MAX_LUGARES]


def fechas_citables(fechas) -> list[str]:
    """Fechas de EVENTO escritas en el texto. La fecha de publicacion no cuenta:
    un informe de 2025 que habla de 2019 describe un hecho de 2019."""
    salida: list[str] = []
    for f in _lista(fechas):
        if isinstance(f, dict) and f.get("origen") == "directo":
            valor = str(f.get("valor") or "").strip()
            if valor and valor not in salida:
                salida.append(valor)
    return salida[:MAX_FECHAS]


def _attr(valor: str) -> str:
    """Valor seguro para un atributo entre comillas simples del bloque <fragmento>."""
    return re.sub(r"[<>']", "", str(valor))


def atributos(fragmento: dict) -> str:
    """Atributos extra del bloque <fragmento>: publicacion, lugares y fechas citables.

    Solo se incluyen los que existen, para no gastar tokens en atributos vacios.
    """
    partes = []
    publicado = re.sub(r"\.0$", "", str(fragmento.get("fecha_publicacion") or "").strip())
    if publicado and publicado.lower() not in ("none", "nan", "nat"):
        partes.append(f"publicado='{_attr(publicado)}'")
    if fragmento.get("lugares_en_texto"):
        partes.append(f"lugares='{_attr('; '.join(fragmento['lugares_en_texto']))}'")
    if fragmento.get("fechas_en_texto"):
        partes.append(f"fechas='{_attr('; '.join(fragmento['fechas_en_texto']))}'")
    return (" " + " ".join(partes)) if partes else ""


#: Reglas comunes que acompanan a los atributos en los prompts del redactor y el
#: comparador. Adaptadas del BASE_REDACTOR del notebook.
REGLAS_LUGARES_FECHAS = (
    "LUGARES: solo puedes atribuir un lugar a una cita si aparece en el atributo `lugares` de "
    "ese fragmento o en su texto; escribelo como aparece ahi. Si el fragmento no trae lugares, "
    "usa su contenido sin atribuirle ubicacion.\n"
    "FECHAS: la fecha de un hecho solo puede salir del atributo `fechas` o del texto del "
    "fragmento. El atributo `publicado` es cuando se publico el documento, NO cuando ocurrio "
    "lo que describe: no lo presentes como fecha del hecho."
)


# ---------------------------------------------------------------------------
# 2. Indices invertidos y deteccion de filtros en la pregunta
# ---------------------------------------------------------------------------
def _iso_pais(iso) -> str:
    """ISO de pais de una entidad. Las subdivisiones pueden venir como `CO-ANT`."""
    return str(iso or "").upper().split("-")[0]


class IndiceGeo:
    """Indices invertidos sobre la metadata enriquecida.

    `por_iso`  : ISO de pais -> filas que mencionan el pais o una de sus subdivisiones.
    `por_anio` : anio de evento -> filas que lo escriben.
    `paises`   : nombre normalizado de un pais -> (canonico, iso), para detectarlo
                 en la pregunta.
    `alias`    : canonico -> formas normalizadas con que aparece en el corpus, para
                 reconocer "Brasil" en una respuesta cuyo fragmento dice "Brazil".
    """

    def __init__(self, geos: Iterable, fechas: Iterable):
        self.por_iso: dict[str, set[int]] = defaultdict(set)
        self.por_anio: dict[int, set[int]] = defaultdict(set)
        self.paises: dict[str, tuple[str, str]] = {}
        self.alias: dict[str, set[str]] = defaultdict(set)

        for i, geo in enumerate(geos):
            for g in _verificadas(geo):
                canon, iso = g.get("canonico"), _iso_pais(g.get("iso"))
                if iso:
                    self.por_iso[iso].add(i)
                if not canon:
                    continue
                for nombre in (g.get("nombre"), canon):
                    clave = normalizar(str(nombre or ""))
                    if len(clave) >= 3:
                        self.alias[canon].add(clave)
                    # Solo paises y solo nombres alfabeticos de 4+ letras: "US" o
                    # "RDC" dispararian con cualquier sigla de la pregunta.
                    if (g.get("clase") == "pais" and iso and len(clave) >= 4
                            and clave.replace(" ", "").isalpha()):
                        self.paises.setdefault(clave, (canon, iso))

        for i, fs in enumerate(fechas):
            for f in _lista(fs):
                if not isinstance(f, dict) or f.get("origen") != "directo":
                    continue
                try:
                    self.por_anio[int(str(f.get("valor", ""))[:4])].add(i)
                except ValueError:
                    continue

        claves = sorted(self.paises, key=len, reverse=True)   # "guinea ecuatorial" antes que "guinea"
        self._rx = (re.compile(r"\b(" + "|".join(map(re.escape, claves)) + r")\b")
                    if claves else None)

    @property
    def vacio(self) -> bool:
        return not self.por_iso and not self.por_anio

    def isos_de(self, paises: Optional[list[str]]) -> list[str]:
        """ISO de una lista de nombres de pais escritos en cualquier variante."""
        salida = []
        for p in paises or []:
            par = self.paises.get(normalizar(str(p)))
            if par and par[1] not in salida:
                salida.append(par[1])
        return salida

    def detectar(self, pregunta: str) -> dict:
        """Paises y rango de anios que la pregunta nombra EXPLICITAMENTE.

        Determinista: un filtro que el usuario no pidio es la forma mas rapida de
        vaciar la evidencia, asi que solo se toma lo que esta escrito.
        """
        plano = normalizar(pregunta)
        paises: list[str] = []
        if self._rx:
            for m in self._rx.finditer(plano):
                canon = self.paises[m.group(1)][0]
                if canon not in paises:
                    paises.append(canon)

        anios = sorted({int(a) for a in re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", plano)})
        desde = hasta = None
        if len(anios) >= 2:
            desde, hasta = anios[0], anios[-1]
        elif anios:
            a = anios[0]
            prefijo = r"\s+(?:el\s+)?(?:ano\s+)?(?:de\s+)?" + str(a)
            if re.search(r"(desde|a partir de|despues de|posterior a)" + prefijo, plano):
                desde = a
            elif re.search(r"(hasta|antes de|previo a)" + prefijo, plano):
                hasta = a
            else:
                desde = hasta = a

        filtros: dict = {}
        if paises:
            filtros["paises"] = paises
        if desde:
            filtros["anio_desde"] = desde
        if hasta:
            filtros["anio_hasta"] = hasta
        return filtros

    def permitidos(self, paises: Optional[list[str]] = None, anio_desde: Optional[int] = None,
                   anio_hasta: Optional[int] = None) -> Optional[set[int]]:
        """Filas que cumplen los filtros; `None` si no hay ningun filtro que aplicar.

        Varios paises se unen (la pregunta puede comparar dos); pais y anio se
        intersecan.
        """
        conjuntos = []
        isos = self.isos_de(paises)
        if isos:
            conjuntos.append(set().union(*(self.por_iso.get(iso, set()) for iso in isos)))
        if anio_desde or anio_hasta:
            anios = range(anio_desde or ANIO_MIN, (anio_hasta or ANIO_MAX) + 1)
            conjuntos.append(set().union(*(self.por_anio.get(a, set()) for a in anios)))
        if not conjuntos:
            return None
        return set.intersection(*conjuntos)


def preferir(ranking: list, permitidos: Optional[set], k: int) -> tuple[list, int]:
    """Los `k` mejores, anteponiendo los que cumplen el filtro.

    Devuelve `(seleccion, n_preferidos)`. Si el filtro no deja `k` candidatos se
    completa con el ranking normal: preferir no puede dejar al redactor sin
    evidencia, que seria peor que no filtrar.
    """
    if permitidos is None:
        return list(ranking[:k]), 0
    dentro = [i for i in ranking if int(i) in permitidos][:k]
    fuera = [i for i in ranking if int(i) not in permitidos][:k - len(dentro)]
    return dentro + fuera, len(dentro)


# ---------------------------------------------------------------------------
# 3. Puntos del mapa de una respuesta de texto
# ---------------------------------------------------------------------------
Corrector = Callable[[dict, dict], Optional[tuple]]


def puntos_de_respuesta(texto: str, fragmentos: list[dict], fuentes: list[dict],
                        alias: Optional[dict] = None, coordenadas: Optional[Corrector] = None,
                        top: int = 25) -> list[dict]:
    """Lugares que la respuesta menciona, tomados de los fragmentos que cita.

    Dos restricciones, como en `datos_para_mapa` del notebook pero a nivel de
    fragmento en vez de documento:
      1. Solo fragmentos citados (sus chunk_id estan en `fuentes`).
      2. Solo lugares cuyo nombre -en cualquiera de sus variantes- aparece en el
         texto final de la respuesta.

    `coordenadas(fragmento, g)` permite corregir homonimos (ver
    visualizador.MotorMetadata.coordenadas); devuelve None para descartar.
    Cada punto conserva `citas`, los indices [n] de la respuesta que lo sustentan.
    """
    indice_de = {cid: f["indice"] for f in fuentes or [] for cid in f.get("chunk_ids", [])}
    if not texto or not indice_de:
        return []
    plano = normalizar(texto)
    menciona: dict[str, bool] = {}

    def aparece(canon: str, g: dict) -> bool:
        if canon not in menciona:
            nombres = {normalizar(str(g.get("nombre") or "")), normalizar(canon)}
            nombres |= (alias or {}).get(canon, set())
            menciona[canon] = any(len(n) >= 3 and re.search(rf"\b{re.escape(n)}\b", plano)
                                  for n in nombres)
        return menciona[canon]

    puntos: dict[str, dict] = {}
    for fr in fragmentos or []:
        cid = str(fr.get("chunk_id"))
        if cid not in indice_de:
            continue
        for g in _verificadas(fr.get("geo")):
            canon = g.get("canonico")
            if not canon or g.get("lat") is None or g.get("lon") is None or not aparece(canon, g):
                continue
            coords = coordenadas(fr, g) if coordenadas else (g["lat"], g["lon"])
            if coords is None:
                continue
            # Homonimo corregido: el ISO original es el del lugar equivocado.
            corregido = tuple(coords) != (g["lat"], g["lon"])
            clave = f"{canon} (corregido)" if corregido else canon
            p = puntos.setdefault(clave, {
                "nombre": str(g.get("nombre") or canon), "entidad": canon,
                "clase": g.get("clase"), "iso": None if corregido else g.get("iso"),
                "lat": float(coords[0]), "lon": float(coords[1]),
                "valor": 0, "doc_ids": [], "citas": [], "_vistos": set()})
            if corregido:
                p["homonimo_corregido"] = True
            if cid in p["_vistos"]:           # un fragmento cuenta una vez por lugar
                continue
            p["_vistos"].add(cid)
            p["valor"] += 1
            doc = str(fr.get("doc_id_oficial") or fr.get("doc_id"))
            if doc not in p["doc_ids"]:
                p["doc_ids"].append(doc)
            if indice_de[cid] not in p["citas"]:
                p["citas"].append(indice_de[cid])

    salida = sorted(puntos.values(), key=lambda p: (-p["valor"], p["nombre"]))[:top]
    for p in salida:
        p.pop("_vistos")
        p["citas"].sort()
    return salida
