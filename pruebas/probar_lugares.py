"""
probar_lugares.py - Lugares y fechas citables, filtro geografico y puntos del mapa.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

No necesita gateway ni base vectorial: `app/lugares.py` y `citas.verificar` son
codigo puro. La metadata se simula con el mismo formato que la enriquecida:

    geo    : [{nombre, canonico, iso, clase, lat, lon, verificado}]
    fechas : [{valor, origen}]

Uso:
    python -m pruebas.probar_lugares
"""
from __future__ import annotations

import sys

from app import citas, lugares

CO = {"nombre": "Colombia", "canonico": "Colombia", "iso": "CO", "clase": "pais",
      "lat": 4.57, "lon": -74.3, "verificado": True}
ANTIOQUIA = {"nombre": "Antioquia", "canonico": "Antioquia", "iso": "CO", "clase": "subdivision",
             "lat": 7.0, "lon": -75.5, "verificado": True}
BRASIL = {"nombre": "Brazil", "canonico": "Brazil", "iso": "BR", "clase": "pais",
          "lat": -10.0, "lon": -55.0, "verificado": True}
BRASIL_ES = dict(BRASIL, nombre="Brasil")
DUDOSO = {"nombre": "Florida", "canonico": "Florida", "iso": "US", "clase": "subdivision",
          "lat": 27.8, "lon": -81.7, "verificado": False}

#: fila i de la "metadata": (geo, fechas)
FILAS = [
    ([CO, ANTIOQUIA], [{"valor": "2019", "origen": "directo"}, {"valor": "2023", "origen": "documento"}]),
    ([ANTIOQUIA], [{"valor": "2021-03", "origen": "directo"}]),
    ([BRASIL], [{"valor": "2019", "origen": "directo"}]),
    ([BRASIL_ES, DUDOSO], []),
    (float("nan"), float("nan")),          # fila sin metadata, como la entrega pandas
]


def _revisar(nombre: str, condicion: bool, detalle: str = "") -> bool:
    print(f"  {'OK  ' if condicion else 'FALLA'} {nombre}{(' - ' + detalle) if detalle else ''}")
    return condicion


def probar_citables() -> bool:
    print("Lugares y fechas citables")
    ok = _revisar("solo verificados, forma literal",
                  lugares.lugares_citables([BRASIL_ES, DUDOSO]) == ["Brasil"])
    ok &= _revisar("solo fechas de evento",
                   lugares.fechas_citables(FILAS[0][1]) == ["2019"])
    ok &= _revisar("fila sin metadata", lugares.lugares_citables(float("nan")) == [])
    frag = {"fecha_publicacion": "2023-04", "lugares_en_texto": ["Colombia", "Antioquia"],
            "fechas_en_texto": ["2019"]}
    attrs = lugares.atributos(frag)
    ok &= _revisar("atributos del bloque <fragmento>",
                   attrs == " publicado='2023-04' lugares='Colombia; Antioquia' fechas='2019'", attrs)
    ok &= _revisar("sin atributos vacios", lugares.atributos({"fecha_publicacion": None}) == "")
    ok &= _revisar("un atributo no rompe el bloque",
                   "'" not in lugares.atributos({"lugares_en_texto": ["Cote d'Ivoire"]})[11:-1])
    return ok


def probar_filtros() -> bool:
    print("Deteccion de filtros e indices")
    ix = lugares.IndiceGeo([g for g, _ in FILAS], [f for _, f in FILAS])
    casos = [
        ("pais en espanol", "¿Qué grupos armados operan en Colombia?", {"paises": ["Colombia"]}),
        ("variante del nombre", "Mineria ilegal en Brasil", {"paises": ["Brazil"]}),
        ("rango de anios", "evolucion entre 2018 y 2022",
         {"anio_desde": 2018, "anio_hasta": 2022}),
        ("desde un anio", "cambios desde 2019", {"anio_desde": 2019}),
        ("anio exacto", "que paso en 2021", {"anio_desde": 2021, "anio_hasta": 2021}),
        ("sin filtros", "riesgos de la basura espacial", {}),
        ("no es subcadena", "la colombianidad", {}),
    ]
    ok = True
    for nombre, pregunta, esperado in casos:
        obtenido = ix.detectar(pregunta)
        ok &= _revisar(nombre, obtenido == esperado, repr(obtenido))
    ok &= _revisar("pais incluye sus subdivisiones",
                   ix.permitidos(["Colombia"]) == {0, 1})
    ok &= _revisar("pais y anio se intersecan",
                   ix.permitidos(["Brasil"], 2019, 2019) == {2})
    ok &= _revisar("sin filtros -> None", ix.permitidos() is None)
    return ok


def probar_preferencia() -> bool:
    print("Preferencia en el ranking")
    sel, n = lugares.preferir([9, 3, 7, 1, 5], {7, 5}, 3)
    ok = _revisar("antepone los que cumplen", sel == [7, 5, 9] and n == 2, repr(sel))
    sel, n = lugares.preferir([9, 3, 7], set(), 3)
    ok &= _revisar("filtro vacio no deja sin evidencia", sel == [9, 3, 7] and n == 0)
    sel, _ = lugares.preferir([9, 3, 7], None, 2)
    ok &= _revisar("sin filtro, ranking intacto", sel == [9, 3])
    return ok


def probar_puntos() -> bool:
    print("Puntos del mapa de una respuesta")
    fragmentos = [
        {"chunk_id": "aaaaaaaa_1", "doc_id": "aaaaaaaa", "doc_id_oficial": "F3-X-001",
         "geo": [CO, ANTIOQUIA]},
        {"chunk_id": "bbbbbbbb_2", "doc_id": "bbbbbbbb", "geo": [BRASIL]},
        {"chunk_id": "cccccccc_3", "doc_id": "cccccccc", "geo": [CO]},       # no citado
    ]
    fuentes = [{"indice": 1, "chunk_ids": ["aaaaaaaa_1"]},
               {"indice": 2, "chunk_ids": ["bbbbbbbb_2"]}]
    alias = {"Brazil": {"brazil", "brasil"}}
    texto = "En Antioquia persisten economias ilegales [1]. En Brasil tambien [2]."
    puntos = lugares.puntos_de_respuesta(texto, fragmentos, fuentes, alias=alias)
    nombres = sorted(p["entidad"] for p in puntos)
    ok = _revisar("solo lugares mencionados y citados", nombres == ["Antioquia", "Brazil"],
                  repr(nombres))
    ant = next(p for p in puntos if p["entidad"] == "Antioquia")
    ok &= _revisar("trazabilidad del punto",
                   ant["doc_ids"] == ["F3-X-001"] and ant["citas"] == [1], repr(ant))
    puntos = lugares.puntos_de_respuesta(texto, fragmentos, fuentes, alias=alias,
                                         coordenadas=lambda f, g: None)
    ok &= _revisar("el corrector puede descartar homonimos", puntos == [])
    ok &= _revisar("sin fuentes no hay mapa",
                   lugares.puntos_de_respuesta(texto, fragmentos, []) == [])
    return ok


def probar_verificacion() -> bool:
    print("Verificacion de citas")
    frags = [{"chunk_id": "13b67ce6657d4fbb_215"}, {"chunk_id": "aa11bb22cc33dd44_12"}]
    v = citas.verificar("Uno [13b67ce6657d4fbb_215]. Dos [ffffffffffffffff_9]. "
                        "Tres [13b67ce6657d4fbb_215].", frags)
    ok = _revisar("cuenta verificadas e inventadas",
                  v["citas_totales"] == 2 and v["citas_verificadas"] == 1
                  and v["citas_descartadas"] == ["ffffffffffffffff_9"]
                  and v["tasa_verificacion"] == 0.5, repr(v))
    v = citas.verificar("Sin citas.", frags)
    ok &= _revisar("sin citas", v["citas_totales"] == 0 and v["tasa_verificacion"] is None)
    return ok


if __name__ == "__main__":
    resultados = [probar_citables(), probar_filtros(), probar_preferencia(),
                  probar_puntos(), probar_verificacion()]
    print("\nTODO OK" if all(resultados) else "\nHAY FALLAS")
    sys.exit(0 if all(resultados) else 1)
