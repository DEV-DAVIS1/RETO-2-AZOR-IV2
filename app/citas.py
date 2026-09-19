"""
citas.py - Conversion de chunk_id en citas legibles.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

El modelo cita por `chunk_id` porque es un identificador exacto que puede copiar
del fragmento sin ambiguedad; pedirle que redacte el nombre de la fuente lo
expondria a inventarlo. Pero `[13b67ce6657d4fbb_178_1]` no le dice nada a un
lector: no permite saber de donde sale la afirmacion sin abrir el JSON.

Este modulo cierra esa brecha en CODIGO, despues de la generacion y sin gastar
tokens: sustituye cada `chunk_id` por un indice numerico agrupado POR DOCUMENTO y
arma la lista de fuentes con los datos reales de la metadata.

    La basura espacial plantea riesgos [13b67ce6657d4fbb_178_1].
    ->
    La basura espacial plantea riesgos [1].

    Fuentes
    [1] UNOOSA. st-space-61rev03s (2017-05) · F2-UNOOSA-030 · fragmentos 178, 215
    [2] CSIS. «Space Threat Assessment 2023» (2023-04) · F2-CSIS-041 · fragmento 12

Se usa el titulo editorial cuando la metadata trae uno real; si no, el nombre
del archivo. Tambien se resuelven los chunk_id que el modelo escribe fuera del
formato pedido: sin corchetes, varios en un corchete o entre parentesis.

La trazabilidad exigida por la seccion 3.3 no se pierde: los `chunk_id` siguen
intactos en `evaluacion.tools_called` y en el panel de evidencia del frontend.

Efecto secundario deseado: una cita que el modelo se invento -un `chunk_id` que
no esta entre los fragmentos recuperados- no tiene fuente que mostrar, asi que se
elimina del texto en lugar de presentarse como respaldo.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from . import config

#: Un chunk_id: <doc_id hexadecimal>_<posicion>, con un sufijo opcional cuando
#: un fragmento largo se partio (`_178_1`).
_ID = r"[0-9a-f]{8,}_\d+(?:_\d+)?"
_ID_COMILLAS = rf"['\"`]?{_ID}['\"`]?"

#: Una cita tal como la pidio el prompt, `[id]`, y las variantes que el modelo
#: produce de todos modos: varias en un corchete (`[id, id]`), corchetes pegados
#: (`[id][id]`) o parentesis (`(id)`). Se tratan como UN grupo para no repetir
#: el mismo indice dentro de el.
PATRON_GRUPO = re.compile(
    rf"[\[(]\s*{_ID_COMILLAS}(?:\s*(?:[,;]|[\])]\s*[\[(])\s*{_ID_COMILLAS})*\s*[\])]")
#: Un chunk_id suelto, sin corchetes. Se resuelve igual: dejarlo crudo en el
#: texto es justo el defecto que este modulo existe para evitar.
PATRON_SUELTO = re.compile(rf"(?<![\w\[])['\"`]?({_ID})['\"`]?(?!\w)")
PATRON_ID = re.compile(_ID)

#: Titulos que la extraccion de la Etapa 1 tomo de las propiedades del archivo
#: y que no describen el documento ("Mapa" en 363 alertas, "untitled"...).
TITULOS_VACIOS = {"mapa", "untitled", "latest updates", "标题", "oea/ser", "presentacion"}
EXTENSIONES = re.compile(r"\.(pdf|docx?|indd|qxd|xlsx?|pptx?|json|csv|txt|html?)$", re.I)
NULOS = ("", "none", "nan", "nat")


def _texto(valor: Any) -> str:
    texto = str(valor if valor is not None else "").strip()
    return "" if texto.lower() in NULOS else texto


def _nativo(valor: Any) -> Any:
    """Tipo nativo de Python. La metadata viene de pandas y un `numpy.int64` en
    `fuentes` hace que FastAPI no pueda serializar la respuesta."""
    return valor.item() if hasattr(valor, "item") else valor


def _nombre_documento(fragmento: dict[str, Any]) -> str:
    """Titulo del documento si la metadata trae uno real; si no, su archivo.

    Dos tercios del corpus tienen titulo editorial ("Space Threat Assessment
    2023"). El resto repite el nombre del archivo o arrastra basura del
    programa que lo genero ("Microsoft Word - X.docx", "X.indd", "Mapa"); en
    esos casos el nombre del archivo, sin el prefijo del observatorio, es la
    referencia mas util que existe.
    """
    observatorio = _observatorio(fragmento)
    archivo = EXTENSIONES.sub("", _texto(fragmento.get("fuente")))
    titulo = re.sub(r"^Microsoft Word - ", "", _texto(fragmento.get("titulo")), flags=re.I)
    era_archivo = bool(EXTENSIONES.search(titulo))
    titulo = EXTENSIONES.sub("", titulo).strip()

    es_editorial = (titulo and titulo != archivo and not era_archivo
                    and titulo.lower() not in TITULOS_VACIOS and len(titulo) >= 8)
    if es_editorial:
        return f"«{titulo}»"

    nombre = archivo or titulo
    if observatorio and nombre.upper().startswith(observatorio.upper() + "_"):
        nombre = nombre[len(observatorio) + 1:]
    return nombre


def _observatorio(fragmento: dict[str, Any]) -> str:
    """Institucion de origen. "OTRO" es la categoria residual de la Etapa 1, no
    una fuente: en ese caso se usa el prefijo del archivo (`INPE_...`, `CEOBS_...`).
    """
    observatorio = _texto(fragmento.get("observatorio"))
    if observatorio.upper() != "OTRO":
        return observatorio
    m = re.match(r"([A-Z][A-Z0-9]{1,14})_", _texto(fragmento.get("fuente")))
    return m.group(1) if m else ""


def _etiqueta(fragmento: dict[str, Any]) -> str:
    """Descripcion legible de un documento, con lo que exista en su metadata.

    Se construye con los campos de la Etapa 1 y degrada con elegancia: un
    documento sin fecha o sin titulo sigue produciendo una cita util.

        CSIS. «Space Threat Assessment 2023» (2023-04)
        UNOOSA. st-space-61rev03s (2017-05)
    """
    partes = [p for p in (_observatorio(fragmento), _nombre_documento(fragmento)) if p]
    etiqueta = ". ".join(partes) or _texto(fragmento.get("doc_id")) or "documento"
    # pandas puede entregar un anio como numero (2017.0) si la columna lo permite.
    fecha = re.sub(r"\.0$", "", _texto(fragmento.get("fecha_publicacion")))
    if fecha:
        etiqueta += f" ({fecha})"
    return etiqueta


def formatear(texto: str, fragmentos: list[dict]) -> tuple[str, list[dict]]:
    """Reemplaza los chunk_id del texto por indices y devuelve las fuentes.

    Los indices se asignan POR DOCUMENTO y en orden de aparicion: varias citas
    del mismo documento comparten numero, que es lo que un lector espera y lo que
    evita una respuesta sembrada de marcas distintas para una sola fuente.

    Devuelve `(texto, fuentes)`, donde cada fuente es:

        {"indice", "etiqueta", "doc_id", "doc_id_oficial", "fuente", "titulo",
         "fecha_publicacion", "observatorio", "formato", "fenomeno",
         "posiciones", "chunk_ids", "referencia"}
    """
    if not texto:
        return texto, []

    por_chunk = {str(f.get("chunk_id")): f for f in fragmentos or []}
    indices: dict[str, int] = {}          # doc_id -> indice de cita
    fuentes: list[dict] = []

    def registrar(chunk_id: str) -> Optional[int]:
        fragmento = por_chunk.get(chunk_id)
        if fragmento is None:
            return None                   # cita inventada: no hay fuente que mostrar
        doc_id = str(fragmento.get("doc_id"))
        if doc_id not in indices:
            indices[doc_id] = len(fuentes) + 1
            fuentes.append({
                "indice": indices[doc_id],
                "etiqueta": _etiqueta(fragmento),
                "doc_id": doc_id,
                "doc_id_oficial": _nativo(fragmento.get("doc_id_oficial")),
                "fuente": _nativo(fragmento.get("fuente")),
                "titulo": _nativo(fragmento.get("titulo")),
                "fecha_publicacion": _nativo(fragmento.get("fecha_publicacion")),
                "observatorio": _nativo(fragmento.get("observatorio")),
                "formato": _nativo(fragmento.get("formato")),
                "fenomeno": _nativo(fragmento.get("fenomeno")),
                "posiciones": [],
                "chunk_ids": [],
            })
        fuente = fuentes[indices[doc_id] - 1]
        if chunk_id not in fuente["chunk_ids"]:
            fuente["chunk_ids"].append(chunk_id)
            posicion = _nativo(fragmento.get("posicion"))
            if posicion is not None and posicion not in fuente["posiciones"]:
                fuente["posiciones"].append(posicion)
        return indices[doc_id]

    def sustituir(m: re.Match) -> str:
        # Un grupo puede traer varios chunk_id; cada documento aparece una vez.
        vistos: list[int] = []
        for chunk_id in PATRON_ID.findall(m.group(0)):
            indice = registrar(chunk_id)
            if indice and indice not in vistos:
                vistos.append(indice)
        return "".join(f"[{i}]" for i in vistos)

    def sustituir_suelto(m: re.Match) -> str:
        indice = registrar(m.group(1))
        return f"[{indice}]" if indice else ""

    texto = PATRON_GRUPO.sub(sustituir, texto)
    texto = PATRON_SUELTO.sub(sustituir_suelto, texto)
    # Cuando no tiene que citar, el modelo a veces rellena el hueco:
    # "[No hay chunk_id relevante]". No es una cita ni le sirve al lector.
    texto = re.sub(r"\s*\[[^\[\]]*chunk_id[^\[\]]*\]", "", texto, flags=re.I)

    # Al quitar citas invalidas o al repetirse un documento pueden quedar marcas
    # duplicadas ("[1] [1]") y espacios sueltos antes de la puntuacion.
    texto = re.sub(r"(\[\d+\])(\s*\1)+", r"\1", texto)
    texto = re.sub(r"[ \t]+([.,;:])", r"\1", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"[ \t]+$", "", texto, flags=re.M)

    for fuente in fuentes:
        fuente["posiciones"].sort(key=lambda p: (isinstance(p, str), p))
        fuente["referencia"] = _referencia(fuente)
    return texto.strip(), fuentes


def _referencia(fuente: dict) -> str:
    """Linea completa de una fuente: etiqueta, id oficial y fragmentos citados."""
    linea = fuente["etiqueta"]
    if fuente.get("doc_id_oficial"):
        linea += f" · {fuente['doc_id_oficial']}"
    if fuente.get("posiciones"):
        marca = "fragmento" if len(fuente["posiciones"]) == 1 else "fragmentos"
        linea += f" · {marca} {', '.join(str(p) for p in fuente['posiciones'])}"
    return linea


def bloque_de_fuentes(fuentes: list[dict]) -> str:
    """Seccion 'Fuentes' en texto plano, para clientes que no leen el campo `fuentes`."""
    if not fuentes:
        return ""
    lineas = ["", "", "**Fuentes**"]
    lineas += [f"[{f['indice']}] {f['referencia']}" for f in fuentes]
    return "\n".join(lineas)


def aplicar(texto: str, fragmentos: list[dict]) -> tuple[str, list[dict]]:
    """Formatea las citas del texto y devuelve la lista de fuentes.

    La lista solo se anexa al texto si `config.FUENTES_EN_TEXTO` lo pide; por
    defecto viaja en el campo `fuentes` y el frontend la muestra bajo la respuesta.
    """
    texto, fuentes = formatear(texto, fragmentos)
    if config.FUENTES_EN_TEXTO:
        texto += bloque_de_fuentes(fuentes)
    return texto, fuentes
