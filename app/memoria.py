"""
memoria.py - Memoria conversacional y registro de trazabilidad.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Corresponde a las dos cajas grises del diagrama de arquitectura: la lectura de
memoria que precede al orquestador y la escritura que cierra cada turno.

Es deliberadamente CODIGO SIN LLM. Resolver una pregunta de seguimiento
("¿y en el fenomeno 2?", "muestrame eso en un mapa") con una llamada adicional al
modelo costaria tokens e interacciones, que son justamente las metricas del
Bloque B (seccion 2.5.2). Aqui se resuelve con un diccionario.

Lo que se recuerda de cada turno es estructurado y acotado a propostio -rama,
fenomenos y doc_ids-, nunca el texto libre del usuario: asi una inyeccion de
prompt no puede sobrevivir a traves de la memoria hasta el turno siguiente.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from typing import Any, Optional

from . import config

log = logging.getLogger("memoria")

_candado = threading.Lock()
#: session_id -> {"turnos": deque, "visto": epoch}. OrderedDict para desalojo LRU.
_sesiones: "OrderedDict[str, dict]" = OrderedDict()


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def nueva_sesion() -> str:
    """Identificador para un cliente que no envio `session_id`."""
    return uuid.uuid4().hex[:16]


def _caducar(ahora: float) -> None:
    """Descarta sesiones inactivas. Se llama con el candado tomado."""
    muertas = [sid for sid, s in _sesiones.items() if ahora - s["visto"] > config.MEMORIA_TTL_S]
    for sid in muertas:
        _sesiones.pop(sid, None)


def leer(session_id: Optional[str]) -> dict:
    """Contexto acumulado de la sesion.

    Devuelve siempre un diccionario con la misma forma, aunque la sesion no exista:

        {"session_id", "turnos": [...], "doc_ids": [...], "fenomenos": [...], "num_turnos"}

    `doc_ids` y `fenomenos` provienen del ultimo turno respondido y son los que
    permiten encadenar preguntas de seguimiento sin volver a preguntar al modelo.
    """
    vacio = {"session_id": session_id, "turnos": [], "doc_ids": [], "fenomenos": [], "num_turnos": 0}
    if not session_id:
        return vacio

    ahora = time.time()
    with _candado:
        _caducar(ahora)
        sesion = _sesiones.get(session_id)
        if not sesion:
            return vacio
        sesion["visto"] = ahora
        _sesiones.move_to_end(session_id)
        turnos = list(sesion["turnos"])

    ultimo = turnos[-1] if turnos else {}
    return {
        "session_id": session_id,
        "turnos": turnos,
        "doc_ids": ultimo.get("doc_ids", []),
        "fenomenos": ultimo.get("fenomenos", []),
        "num_turnos": len(turnos),
    }


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
def escribir(session_id: str, pregunta: str, rama: str, fenomenos: list[int],
             doc_ids: list[str], estado: str = "ok") -> None:
    """Registra el turno recien resuelto en la sesion.

    De la pregunta solo se guarda un recorte corto para depuracion; lo que se
    reutiliza en el turno siguiente son los campos estructurados.
    """
    if not session_id:
        return
    turno = {
        "ts": time.time(),
        "pregunta": (pregunta or "")[:200],
        "rama": rama,
        "fenomenos": list(fenomenos or []),
        "doc_ids": list(doc_ids or [])[:20],
        "estado": estado,
    }
    ahora = time.time()
    with _candado:
        _caducar(ahora)
        sesion = _sesiones.get(session_id)
        if sesion is None:
            sesion = {"turnos": deque(maxlen=config.MEMORIA_TURNOS), "visto": ahora}
            _sesiones[session_id] = sesion
        sesion["turnos"].append(turno)
        sesion["visto"] = ahora
        _sesiones.move_to_end(session_id)
        while len(_sesiones) > config.MEMORIA_SESIONES:
            _sesiones.popitem(last=False)          # desaloja la sesion mas antigua


def registrar(evento: dict[str, Any]) -> None:
    """Anexa una linea al log JSONL de interacciones.

    El log es la traza auditable de la solucion: una linea por consulta con rama,
    agentes invocados, tokens, latencia y doc_ids. Un fallo de escritura nunca
    debe tumbar una respuesta ya generada, por eso el error solo se registra.
    """
    try:
        config.RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with config.RUTA_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        log.warning("No se pudo escribir el log de interacciones: %s", e)


# ---------------------------------------------------------------------------
# Reutilizacion de contexto (codigo, sin LLM)
# ---------------------------------------------------------------------------
#: Marcas de que la pregunta se apoya en el turno anterior en vez de abrir un tema.
_SEGUIMIENTO = (
    "eso", "esos", "esas", "esto", "estos", "ahi", "alli", "lo anterior", "lo mismo",
    "esa respuesta", "ese dato", "los anteriores", "las anteriores", "de ahi",
    "profundiza", "amplia", "y ahora", "y en", "tambien", "mismo",
)


def es_seguimiento(pregunta: str) -> bool:
    """Heuristica: la pregunta referencia el turno anterior en lugar de abrir un tema."""
    from .seguridad import normalizar
    plano = normalizar(pregunta)
    if len(plano.split()) <= 4:                 # "¿y en un mapa?", "amplia eso"
        return True
    return any(f" {m} " in f" {plano} " for m in _SEGUIMIENTO)


def heredar_fenomenos(fenomenos: list[int], contexto: dict, pregunta: str) -> list[int]:
    """Completa el fenomeno cuando la pregunta de seguimiento lo da por sobrentendido.

    "¿Que pasa con la basura espacial?" seguida de "muestrame eso en un mapa": el
    orquestador no ve el fenomeno en la segunda pregunta, pero la memoria si.
    """
    if fenomenos or not contexto.get("fenomenos"):
        return fenomenos
    return list(contexto["fenomenos"]) if es_seguimiento(pregunta) else fenomenos
