"""
seguridad.py - Filtro determinista de entrada (Bloque C de la evaluacion, seccion 2.5.3).
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Primera de las tres barreras de defensa del sistema:

  1. Este filtro, en codigo y SIN LLM: bloquea la consulta antes de gastar un solo
     token. Un ataque detectado aqui cuesta 0 tokens y ~1 ms de latencia.
  2. El orquestador, que clasifica como `fuera_dominio` cualquier intento de cambio
     de rol que sobreviva al paso 1 (app/agentes/orquestador.py).
  3. Los prompts de redactor, comparador y visualizador, que tratan el texto de los
     fragmentos recuperados como DATOS y no como instrucciones, para cerrar la via
     de inyeccion indirecta a traves del corpus.

Criterio de diseno: el filtro solo bloquea patrones de alta confianza. Una pregunta
legitima mal clasificada costaria puntos en el Bloque A (calidad), que pesa el doble
que el Bloque C; ante la duda, se deja pasar y decide el orquestador.
"""
from __future__ import annotations

import re
import unicodedata

#: Longitud maxima aceptada. Las inyecciones suelen venir en muros de texto que
#: intentan empujar el system prompt fuera de la ventana de contexto.
MAX_CARACTERES = 2000

_MOTIVOS = {
    "cambio_de_instrucciones": r"""
        (ignor\w*|olvid\w*|descart\w*|desatend\w*|omit\w*|anul\w*)\s+
        ((tod\w+|las?|tus?|sus?|los?|el|la)\s+)*
        (instruccion|indicacion|regla|directriz|orden|prompt|lineamiento|restriccion|politica)
      | (ignore|disregard|forget|override|bypass)\s+((all|any|your|the|previous|prior)\s+)*
        (instruction|rule|prompt|direction|guideline|restriction|polic)
      | (instruccion|regla|prompt)\w*\s+(anterior|previa|original)\w*\s+(ya\s+)?no\s+(aplic|val|cuent)
      | nuevas?\s+instruccion\w*\s*[:\-]
      | new\s+instructions?\s*[:\-]
    """,
    "exfiltracion_de_prompt": r"""
        (muestr\w*|repit\w*|imprim\w*|revel\w*|dime|dinos|comparte|transcrib\w*|escrib\w*|copia)
        \s+((me|nos|todo|exactamente|literal\w*|palabra\s+por\s+palabra)\s+)*
        ((tus?|sus?|el|las?|los)\s+)*
        (system\s*prompt|prompt\s+de\s+sistema|prompt\s+del?\s+sistema|prompt\s+inicial
         |instruccion\w*\s+(de\s+)?(sistema|internas?|originales?|completas?)
         |configuracion\s+interna|reglas?\s+(internas?|de\s+sistema))
      | (print|output|show|reveal|repeat|display|echo)\s+((me|us|the|your|all)\s+)*
        (system\s*prompt|initial\s+prompt|instructions?\s+above|text\s+above|prompt\s+verbatim)
      | que\s+(dice|contiene|hay\s+en)\s+(tu|su)\s+(system\s*prompt|prompt|configuracion)
      | repeat\s+(everything|the\s+words)\s+above
    """,
    "suplantacion_de_rol": r"""
        (a\s+partir\s+de\s+ahora|desde\s+ahora|de\s+ahora\s+en\s+adelante|a\s+partir\s+de\s+este\s+momento)
        \s+(eres|seras|actuaras|te\s+comportaras|responde\w*|vas\s+a\s+ser)
      | (actua|comportate|responde|haz\s+de\s+cuenta|finge|simula|pretende)\s+como\s+si\s+(fueras|no)
      | (eres|seras)\s+(ahora\s+)?(un|una|el|la)\s+\w+\s+(sin|que\s+no\s+tiene)\s+(restriccion|limit|filtro|regla)
      | you\s+are\s+(now|no\s+longer)\s+
      | (act|behave|respond)\s+as\s+(if\s+you|an?\s+unrestricted|dan\b)
      | \bdan\s+mode\b | \bjailbreak | modo\s+(desarrollador|dios|libre|sin\s+filtros)
      | developer\s+mode | \bsudo\s+mode\b
      | sin\s+(ninguna\s+)?(restriccion|censura|filtro|limitacion)\w*\s+(alguna|de\s+ningun)
    """,
    "exfiltracion_de_credenciales": r"""
        (api[\s_-]?key|llave|credencial|token\s+de\s+acceso|contrasena|password|secret|
         variable\w*\s+de\s+entorno|environment\s+variable|\.env\b|os\.environ)
        [^.?!]{0,40}
        (cual|cuales|dime|muestr|imprim|revel|comparte|lista|dame|what|show|print|reveal|give)
      | (cual|dime|muestr\w*|imprim\w*|revel\w*|dame|lista\w*)
        [^.?!]{0,40}
        (api[\s_-]?key|llave\s+de\s+api|credencial|contrasena|password|secret\s+key|
         variable\w*\s+de\s+entorno|environment\s+variable|\.env\b)
    """,
    "ejecucion_de_codigo": r"""
        (ejecuta|corre|run|execute|eval)\s+(este|el\s+siguiente|this|the\s+following)?\s*
        (codigo|comando|script|code|command|shell|bash|python)
      | rm\s+-rf\s+/ | drop\s+table | ;\s*shutdown | subprocess\.|__import__
    """,
}

_PATRONES = {
    motivo: re.compile(expr, re.IGNORECASE | re.VERBOSE)
    for motivo, expr in _MOTIVOS.items()
}

#: Los mismos patrones con los espacios obligatorios relajados. Se usan solo sobre
#: el texto compactado, donde ya no quedan separadores que consumir.
_PATRONES_COMPACTOS = {
    motivo: re.compile(expr.replace(r"\s+", r"\s*"), re.IGNORECASE | re.VERBOSE)
    for motivo, expr in _MOTIVOS.items()
}

#: Una racha de letras sueltas ("i g n o r a") es la evasion tipografica clasica.
#: Detectarla antes de compactar evita compactar texto normal, donde unir palabras
#: produciria coincidencias falsas.
_EVASION_ESPACIADA = re.compile(r"(?:\b\w\b[\s.\-_*]+){4,}\w")

#: Respuesta unica ante cualquier ataque. No repite el texto del atacante (evita que
#: el ataque se refleje en `respuesta`) ni describe que regla se activo (no da pistas
#: para afinar el siguiente intento).
MENSAJE_BLOQUEO = (
    "No puedo atender esa solicitud. Este asistente esta limitado al analisis documental "
    "de tres fenomenos: inteligencia artificial en entornos militares, seguridad espacial "
    "y orbita baja terrestre, y dinamicas territoriales en America Latina. "
    "Formula tu pregunta sobre alguno de ellos y la respondo con gusto."
)


def normalizar(texto: str) -> str:
    """Minusculas, sin tildes, sin caracteres invisibles y con espacios colapsados.

    Neutraliza `IGNORA`, `ignóra` y los caracteres de control con los que se
    intenta esconder texto (`\\u200b`, marcas de direccion bidireccional).
    """
    plano = unicodedata.normalize("NFKD", texto or "").lower()
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    plano = re.sub(r"[​-‏‪-‮﻿]", "", plano)   # controles invisibles
    return re.sub(r"\s+", " ", plano).strip()


def _compactar(plano: str) -> str:
    """Une las letras de una consulta escrita espaciada: `i g n o r a` -> `ignora`."""
    return re.sub(r"[\s.\-_*]+", "", plano)


def revisar(pregunta: str) -> tuple[bool, str]:
    """Evalua la consulta del usuario.

    Devuelve `(es_segura, motivo)`. Con `es_segura=False` el servidor responde
    MENSAJE_BLOQUEO sin invocar a ningun modelo.
    """
    if not pregunta or not pregunta.strip():
        return False, "consulta_vacia"
    if len(pregunta) > MAX_CARACTERES:
        return False, "consulta_demasiado_larga"

    plano = normalizar(pregunta)
    for motivo, patron in _PATRONES.items():
        if patron.search(plano):
            return False, motivo

    # Segunda pasada solo si la consulta viene escrita letra por letra.
    if _EVASION_ESPACIADA.search(plano):
        compacto = _compactar(plano)
        for motivo, patron in _PATRONES_COMPACTOS.items():
            if patron.search(compacto):
                return False, f"{motivo}_espaciado"
    return True, "ok"


def sanear_fragmento(texto: str) -> str:
    """Neutraliza delimitadores de rol dentro del texto recuperado del corpus.

    Defensa contra inyeccion indirecta: si un documento del corpus contiene
    `<|im_start|>system` o `### Instruction:`, el modelo podria interpretarlo como
    un turno real de la conversacion. Se desactivan esos marcadores conservando el
    contenido legible, porque el fragmento sigue siendo evidencia citable.
    """
    limpio = re.sub(r"<\|[^|>]{0,40}\|>", " ", texto or "")
    limpio = re.sub(r"(?im)^\s*#{2,}\s*(instruction|system|assistant|user)\b.*$", " ", limpio)
    limpio = re.sub(r"(?i)\b(system|assistant|user)\s*:\s*(?=\n)", r"\1. ", limpio)
    return limpio
