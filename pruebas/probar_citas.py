"""
probar_citas.py - Verifica la conversion de chunk_id en citas legibles.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

No necesita gateway ni base vectorial: `app/citas.py` es codigo puro. Cubre los
formatos en que el modelo escribe un chunk_id aunque el prompt pida `[id]`, el
descarte de citas inventadas y la etiqueta que se arma con la metadata.

Uso:
    python -m pruebas.probar_citas
"""
from __future__ import annotations

import re
import sys

from app import citas

UNOOSA = {"doc_id": "13b67ce6657d4fbb", "fuente": "UNOOSA_st-space-61rev03s.pdf",
          "titulo": "UNOOSA_st-space-61rev03s", "observatorio": "UNOOSA",
          "fecha_publicacion": "2017-05", "doc_id_oficial": "F2-UNOOSA-030", "fenomeno": 2}
CSIS = {"doc_id": "aa11bb22cc33dd44", "fuente": "CSIS_space-threat-2023.pdf",
        "titulo": "Space Threat Assessment 2023", "observatorio": "CSIS",
        "fecha_publicacion": "2023-04", "doc_id_oficial": "F2-CSIS-041", "fenomeno": 2}
FRAGMENTOS = [
    dict(UNOOSA, chunk_id="13b67ce6657d4fbb_178_1", posicion=178),
    dict(UNOOSA, chunk_id="13b67ce6657d4fbb_215", posicion=215),
    dict(CSIS, chunk_id="aa11bb22cc33dd44_12", posicion=12),
]
CHUNK_CRUDO = re.compile(r"[0-9a-f]{8,}_\d+")


def _revisar(nombre: str, condicion: bool, detalle: str = "") -> bool:
    print(f"  {'OK  ' if condicion else 'FALLA'} {nombre}{(' - ' + detalle) if detalle else ''}")
    return condicion


def probar_formatos() -> bool:
    """Ningun chunk_id debe sobrevivir en el texto, venga como venga."""
    print("Formatos de cita")
    casos = [
        ("corchetes", "Riesgo [13b67ce6657d4fbb_178_1].", "Riesgo [1]."),
        ("sin corchetes", "Riesgo 13b67ce6657d4fbb_215.", "Riesgo [1]."),
        ("parentesis", "Riesgo (13b67ce6657d4fbb_215).", "Riesgo [1]."),
        ("lista, mismo documento", "Riesgo [13b67ce6657d4fbb_178_1, 13b67ce6657d4fbb_215].",
         "Riesgo [1]."),
        ("lista, dos documentos", "Riesgo [13b67ce6657d4fbb_215; aa11bb22cc33dd44_12].",
         "Riesgo [1][2]."),
        ("corchetes pegados", "Riesgo [13b67ce6657d4fbb_215][13b67ce6657d4fbb_178_1].",
         "Riesgo [1]."),
        ("cita inventada", "Riesgo [ffffffffffffffff_9].", "Riesgo."),
        ("relleno sin cita", "No hay datos. [No hay chunk_id relevante]", "No hay datos."),
    ]
    ok = True
    for nombre, entrada, esperado in casos:
        texto, _ = citas.formatear(entrada, FRAGMENTOS)
        ok &= _revisar(nombre, texto == esperado and not CHUNK_CRUDO.search(texto), repr(texto))
    return ok


def probar_fuentes() -> bool:
    """La lista de fuentes sale de la metadata, agrupada por documento."""
    print("Lista de fuentes")
    texto, fuentes = citas.aplicar(
        "Uno [13b67ce6657d4fbb_215]. Dos [aa11bb22cc33dd44_12]. Tres [13b67ce6657d4fbb_178_1].",
        FRAGMENTOS)
    ok = _revisar("un indice por documento", [f["indice"] for f in fuentes] == [1, 2])
    ok &= _revisar("archivo sin prefijo repetido",
                   fuentes[0]["etiqueta"] == "UNOOSA. st-space-61rev03s (2017-05)",
                   fuentes[0]["etiqueta"])
    ok &= _revisar("titulo editorial",
                   fuentes[1]["etiqueta"] == "CSIS. «Space Threat Assessment 2023» (2023-04)",
                   fuentes[1]["etiqueta"])
    ok &= _revisar("fragmentos y chunk_id trazables",
                   fuentes[0]["posiciones"] == [178, 215] and len(fuentes[0]["chunk_ids"]) == 2)
    ok &= _revisar("referencia con id oficial y fragmentos",
                   fuentes[0]["referencia"].endswith("F2-UNOOSA-030 · fragmentos 178, 215"),
                   fuentes[0]["referencia"])
    # `respuesta` es tambien `actual_output`: por defecto no carga metadata ajena
    # al contexto recuperado. La lista viaja en el campo `fuentes`.
    ok &= _revisar("el texto no lleva el bloque por defecto", "Fuentes" not in texto, repr(texto))
    ok &= _revisar("bloque disponible en texto plano",
                   len(citas.bloque_de_fuentes(fuentes).splitlines()) == 5)
    return ok


def probar_titulos() -> bool:
    """Titulos heredados del programa que genero el archivo no son titulos."""
    print("Titulos de la metadata")
    casos = [
        ({"observatorio": "ALERTAS", "fuente": "ALERTAS_031-23-91889.json", "titulo": "Mapa"},
         "ALERTAS. 031-23-91889"),
        ({"observatorio": "CSIS", "fuente": "CSIS_testimony.pdf",
          "titulo": "Microsoft Word - Testimony to House.docx"}, "CSIS. testimony"),
        ({"observatorio": "OTRO", "fuente": "CEOBS_catalog-2.json", "titulo": None,
          "fecha_publicacion": "2026"}, "CEOBS. catalog-2 (2026)"),
        ({"doc_id": "abc123"}, "abc123"),
    ]
    ok = True
    for fragmento, esperado in casos:
        etiqueta = citas._etiqueta(fragmento)
        ok &= _revisar(esperado, etiqueta == esperado, etiqueta)
    return ok


def probar_serializacion() -> bool:
    """`fuentes` viaja en la respuesta HTTP: debe sobrevivir al encoder de FastAPI.

    La metadata se lee con pandas, que entrega `numpy.int64`; si ese tipo llega a
    `fuentes`, el endpoint responde 500 solo cuando la respuesta trae citas.
    """
    print("Serializacion con tipos de pandas")
    import pandas as pd
    from fastapi.encoders import jsonable_encoder

    fila = pd.DataFrame([dict(UNOOSA, chunk_id="13b67ce6657d4fbb_215", posicion=215,
                              formato="pdf")]).iloc[0]
    # Acceso por campo, como en buscar_corpus: `to_dict()` ya convertiria a int.
    fragmento = {campo: fila[campo] for campo in fila.index}
    ok = _revisar("la fila trae numpy.int64", type(fragmento["posicion"]).__name__ == "int64")
    if not ok:
        return False
    _, fuentes = citas.aplicar("Riesgo [13b67ce6657d4fbb_215].", [fragmento])
    try:
        jsonable_encoder({"fuentes": fuentes})
        return _revisar("fuentes serializa a JSON", True)
    except ValueError as e:
        return _revisar("fuentes serializa a JSON", False, str(e)[:120])


if __name__ == "__main__":
    resultados = [probar_formatos(), probar_fuentes(), probar_titulos()]
    resultados.append(probar_serializacion())
    print("\nTODO OK" if all(resultados) else "\nHAY FALLAS")
    sys.exit(0 if all(resultados) else 1)
