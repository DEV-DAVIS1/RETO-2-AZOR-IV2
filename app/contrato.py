"""
contrato.py - Construccion del JSON de respuesta de la seccion 2.4.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Todas las ramas del sistema -redactor, comparador, visualizador, respuesta fija y
errores- salen por aqui, de modo que el endpoint devuelve siempre la misma
estructura de tres bloques: `respuesta`, `evaluacion` y `metadata`.

Requisito obligatorio de la seccion 2.4: `metadata.tokens.total` debe reflejar el
consumo de TODOS los modelos que participaron en la respuesta, no solo el del
orquestador. `Contador` existe para garantizarlo: cada agente anota su consumo y
el total se deriva de esas anotaciones, nunca se escribe a mano.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from . import config


class Contador:
    """Acumulador de consumo por agente durante un turno.

    Cada llamada a un modelo se anota con `anotar()`; de ahi salen tanto
    `metadata.tokens` como `metadata.tokens_por_agente` y `num_interacciones`.
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self._por_agente: list[dict[str, Any]] = []
        self._orden: list[str] = []

    def anotar(self, agente: str, modelo: str, entrada: int, salida: int,
               llamadas: int = 1) -> None:
        """Registra el consumo de un agente.

        `llamadas` permite anotar de una vez varias invocaciones al modelo: el
        visualizador, por ejemplo, llama dos veces (seleccion y descripcion) y
        reporta el consumo agregado.
        """
        entrada, salida = int(entrada or 0), int(salida or 0)
        for fila in self._por_agente:
            if fila["agente"] == agente and fila["modelo"] == modelo:
                fila["input"] += entrada
                fila["output"] += salida
                fila["total"] += entrada + salida
                fila["llamadas"] += llamadas
                break
        else:
            self._por_agente.append({"agente": agente, "modelo": modelo, "input": entrada,
                                     "output": salida, "total": entrada + salida,
                                     "llamadas": llamadas})
        self.invocar(agente)

    def invocar(self, agente: str) -> None:
        """Deja constancia de que un agente participo, aunque no gaste tokens.

        El caso real es el visualizador: la generacion del componente es
        determinista y puede resolverse por heuristica sin llamar al modelo.
        """
        if agente not in self._orden:
            self._orden.append(agente)

    @property
    def agentes(self) -> list[str]:
        return list(self._orden)

    @property
    def num_interacciones(self) -> int:
        """Numero de llamadas a modelos realizadas para resolver la consulta."""
        return sum(f["llamadas"] for f in self._por_agente)

    def totales(self) -> dict[str, int]:
        entrada = sum(f["input"] for f in self._por_agente)
        salida = sum(f["output"] for f in self._por_agente)
        return {"input": entrada, "output": salida, "total": entrada + salida}

    def por_agente(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in f.items() if k != "llamadas"} for f in self._por_agente]

    def latencia_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)


def construir(pregunta: str, respuesta: str, contador: Contador, *,
              evaluacion: Optional[dict] = None, estado: str = "ok",
              rama: str = "", fenomenos: Optional[list[int]] = None,
              visualizacion: Optional[dict] = None, fuentes: Optional[list] = None,
              session_id: Optional[str] = None) -> dict:
    """Arma la respuesta completa del endpoint.

    `evaluacion` es lo que devolvio el agente de la rama. Si la rama no recupero
    nada -respuesta fija fuera de dominio, bloqueo de seguridad, error- se omiten
    `retrieval_context` y `tools_called`, tal como permite la restriccion de la
    seccion 2.4 ("obligatorios cuando apliquen").

    `visualizacion` y `traza` son extensiones propias para el frontend y el
    dashboard del Reto 2; no alteran los tres bloques que evalua ADL.
    """
    ev = dict(evaluacion or {})
    ev["input"] = pregunta
    ev["actual_output"] = respuesta
    if not ev.get("retrieval_context"):
        ev.pop("retrieval_context", None)
    if not ev.get("tools_called"):
        ev.pop("tools_called", None)

    salida = {
        "respuesta": respuesta,
        "evaluacion": ev,
        "metadata": {
            "num_interacciones": contador.num_interacciones,
            "agentes_invocados": contador.agentes,
            "tokens": contador.totales(),
            "tokens_por_agente": contador.por_agente(),
            "latencia_ms": contador.latencia_ms(),
            "estado": estado,
        },
        # Documentos que sustentan la respuesta, ya resueltos a un nombre legible.
        # Las citas [1], [2]... del texto apuntan al `indice` de esta lista.
        "fuentes": fuentes or [],
        "traza": {
            "rama": rama,
            "fenomenos": fenomenos or [],
            "session_id": session_id,
        },
    }
    if visualizacion is not None:
        salida["visualizacion"] = visualizacion
    return salida


def doc_ids_de(evaluacion: dict) -> list[str]:
    """Extrae los doc_id citados, para que la memoria los ofrezca al turno siguiente.

    Las herramientas devuelven identificadores en formato `<doc_id>:<chunk_id>`
    (buscar_corpus) o `<doc_id>` suelto (consultar_datos).
    """
    vistos: list[str] = []
    for llamada in (evaluacion or {}).get("tools_called", []):
        for ident in llamada.get("output") or []:
            doc_id = str(ident).partition(":")[0]
            if doc_id and doc_id not in vistos:
                vistos.append(doc_id)
    return vistos[:20]


def evidencia_de(evaluacion: dict, fen_por_chunk: Optional[dict] = None) -> list[dict]:
    """Empareja `retrieval_context` con los ids de `tools_called`.

    Da al frontend la evidencia trazable exigida por la seccion 3.3: cada texto
    mostrado queda ligado a su doc_id y su chunk_id de origen.
    """
    textos = (evaluacion or {}).get("retrieval_context") or []
    ids = [i for t in (evaluacion or {}).get("tools_called", []) for i in (t.get("output") or [])]
    evidencia = []
    for texto, ident in zip(textos, ids):
        doc_id, _, chunk_id = str(ident).partition(":")
        evidencia.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "fenomeno": (fen_por_chunk or {}).get(chunk_id),
            "texto": " ".join(str(texto).split())[:700],
        })
    return evidencia


def error(pregunta: str, contador: Contador, estado: str, *,
          rama: str = "", session_id: Optional[str] = None) -> dict:
    """Respuesta degradada pero valida cuando una rama falla.

    El contrato se mantiene siempre: ADL debe poder parsear la respuesta incluso
    cuando el gateway de modelos no esta disponible.
    """
    return construir(
        pregunta,
        "En este momento no puedo completar el analisis por un problema tecnico del "
        "servicio de modelos. Intenta de nuevo en unos segundos.",
        contador, estado=estado, rama=rama or "error", session_id=session_id)


def fuera_de_dominio(pregunta: str, contador: Contador, *,
                     mensaje: str = config.MENSAJE_FUERA_DOMINIO,
                     estado: str = "ok", rama: str = "fuera_dominio",
                     session_id: Optional[str] = None) -> dict:
    """Respuesta fija. Cuesta cero tokens de generacion y cierra el turno."""
    return construir(pregunta, mensaje, contador, estado=estado, rama=rama,
                     fenomenos=[], session_id=session_id)
