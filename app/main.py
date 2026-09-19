"""
main.py - API HTTP del sistema multi-agente.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Implementa el flujo completo del diagrama de arquitectura:

    pregunta
       |
       v
    memoria (lectura) + filtro de seguridad     <- codigo, sin LLM
       |
       v
    1. ORQUESTADOR (gpt-oss-20b)  --- tool: buscar_corpus
       |
       +--> respuesta fija (fuera de dominio)   <- codigo, sin LLM
       +--> 2. REDACTOR      (incluye cronologia)
       +--> 3. COMPARADOR    (2-3 fenomenos)
       +--> 4. VISUALIZADOR  (graficos y mapas) --- tool: consultar_datos
       |
       v
    memoria (escritura) + log JSONL             <- codigo, sin LLM
       |
       v
    respuesta JSON (seccion 2.4)

Endpoints:
    POST /chat        contrato oficial que evalua ADL (seccion 2.4)
    GET  /agent-card  ficha del sistema multi-agente (seccion 2.3)
    GET  /salud       healthcheck del contenedor (Anexo A.4)
    POST /consulta    misma respuesta enriquecida para el dashboard del Reto 2
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAIError
from pydantic import BaseModel, Field

from . import config, contrato, memoria, seguridad
from .agentes import comparador, herramientas, orquestador, redactor, visualizador

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("main")

app = FastAPI(
    title="CODEFEST AD ASTRA 2026 - Asistente multi-agente",
    description="Equipo Azor IV - Reto 1: asistente conversacional sobre el corpus de la Etapa 1.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

RAIZ = config.RAIZ
FICHA = RAIZ / "agent_card.json"


class Consulta(BaseModel):
    """Entrada del endpoint. Solo `pregunta` es obligatoria."""

    pregunta: str = Field(..., description="Pregunta del usuario en lenguaje natural.")
    session_id: Optional[str] = Field(None, description="Hilo conversacional; opcional.")
    fenomeno: Optional[int] = Field(None, ge=1, le=3,
                                    description="Fuerza el fenomeno en lugar de inferirlo.")
    doc_ids: Optional[list[str]] = Field(None, description="Documentos de contexto del turno previo.")


# ---------------------------------------------------------------------------
# Ramas especializadas
# ---------------------------------------------------------------------------
def _rama_redactor(c: Consulta, decision: dict, contador) -> dict[str, Any]:
    fenomeno = decision["fenomenos"][0] if len(decision["fenomenos"]) == 1 else None
    res = redactor.responder(c.pregunta, contador, fenomeno=fenomeno, query=decision["query"])
    return {"respuesta": res["respuesta"], "evaluacion": res["evaluacion"],
            "fuentes": res["fuentes"],
            "evidencia": contrato.evidencia_de(res["evaluacion"])}


def _rama_comparador(c: Consulta, decision: dict, contador) -> dict[str, Any]:
    res = comparador.comparar(c.pregunta, decision["fenomenos"], decision["query"], contador)
    return {"respuesta": res["respuesta"], "evaluacion": res["evaluacion"],
            "estado": res["estado"], "fuentes": res["fuentes"],
            "evidencia": contrato.evidencia_de(res["evaluacion"], res["fen_por_chunk"])}


def _rama_visualizador(c: Consulta, decision: dict, contador,
                       doc_ids_previos: list[str]) -> dict[str, Any]:
    res = visualizador.visualizar(
        c.pregunta,
        fenomenos=decision["fenomenos"],
        contexto_sesion={"doc_ids": c.doc_ids or doc_ids_previos},
        buscador_tema=herramientas.buscador_tema,
        contador=contador,
    )
    viz = res["visualizacion"]
    salida = {"respuesta": res["respuesta"], "evaluacion": res["evaluacion"],
              "estado": res["metadata"].get("estado", "ok"), "visualizacion": viz,
              "puntos": [], "grafico": None, "evidencia": []}

    componente, datos = viz["componente"], viz.get("datos") or []
    if componente in ("mapa_puntos", "mapa_coropletico"):
        salida["puntos"] = [
            {"nombre": d.get("etiqueta") or d.get("entidad"), "lat": d["lat"], "lon": d["lon"],
             "valor": d.get("valor", 0), "doc_ids": d.get("doc_ids", [])}
            for d in datos if d.get("lat") is not None and d.get("lon") is not None]
        salida["grafico"] = {"componente": componente, "titulo": viz["titulo"], "datos": [],
                             "explicacion": viz.get("explicacion", "")}
    elif componente == "panel_evidencia":
        salida["evidencia"] = [
            {"doc_id": d["doc_id"], "chunk_id": d.get("chunk_id"),
             "fenomeno": d.get("fenomeno"), "texto": d.get("texto", "")} for d in datos]
    else:   # linea_tiempo, matriz_calor, red_coocurrencia
        salida["grafico"] = {"componente": componente, "titulo": viz["titulo"], "datos": datos,
                             "explicacion": viz.get("explicacion", "")}
    return salida


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def resolver(c: Consulta) -> dict[str, Any]:
    """Resuelve una consulta de principio a fin y devuelve el contrato completo."""
    contador = contrato.Contador()
    session_id = c.session_id or memoria.nueva_sesion()

    # --- Memoria (lectura) + filtro -------------------------------------
    # Ambos son codigo puro: un ataque o una pregunta vacia se resuelven sin
    # gastar un token y sin latencia de red.
    contexto = memoria.leer(c.session_id)
    segura, motivo = seguridad.revisar(c.pregunta)
    if not segura:
        log.warning("consulta bloqueada por el filtro: %s", motivo)
        salida = contrato.fuera_de_dominio(
            c.pregunta, contador, mensaje=seguridad.MENSAJE_BLOQUEO,
            estado="ok", rama="bloqueado", session_id=session_id)
        _cerrar(salida, session_id, c.pregunta, "bloqueado", [], [], motivo)
        return salida

    # --- 1. Orquestador --------------------------------------------------
    try:
        decision = orquestador.orquestar(c.pregunta, contador)
    except OpenAIError as e:
        log.exception("fallo el orquestador")
        salida = contrato.error(c.pregunta, contador, f"error_orquestador:{type(e).__name__}",
                                session_id=session_id)
        _cerrar(salida, session_id, c.pregunta, "error", [], [], "error_orquestador")
        return salida

    # El fenomeno explicito del cliente manda; si no hay, la memoria completa el
    # que la pregunta de seguimiento da por sobrentendido.
    if c.fenomeno in (1, 2, 3):
        decision["fenomenos"] = [c.fenomeno]
    else:
        decision["fenomenos"] = memoria.heredar_fenomenos(
            decision["fenomenos"], contexto, c.pregunta)
    rama = decision["rama"]

    # --- 2-4. Rama especializada -----------------------------------------
    try:
        if rama == "fuera_dominio":
            salida = contrato.fuera_de_dominio(c.pregunta, contador, session_id=session_id)
            _cerrar(salida, session_id, c.pregunta, rama, [], [], "ok")
            return salida
        if rama == "comparador" and len(decision["fenomenos"]) >= 2:
            res = _rama_comparador(c, decision, contador)
        elif rama == "visualizador":
            res = _rama_visualizador(c, decision, contador, contexto["doc_ids"])
        else:
            rama = "redactor"
            res = _rama_redactor(c, decision, contador)
    except OpenAIError as e:
        log.exception("fallo la rama %s", rama)
        salida = contrato.error(c.pregunta, contador, f"error_llm:{type(e).__name__}",
                                rama=rama, session_id=session_id)
        _cerrar(salida, session_id, c.pregunta, rama, decision["fenomenos"], [], "error_llm")
        return salida

    salida = contrato.construir(
        c.pregunta, res["respuesta"], contador,
        evaluacion=res["evaluacion"], estado=res.get("estado", "ok"),
        rama=rama, fenomenos=decision["fenomenos"],
        visualizacion=res.get("visualizacion"), fuentes=res.get("fuentes"),
        session_id=session_id)

    # Extensiones para el frontend y el dashboard del Reto 2. Viven fuera de los
    # tres bloques que evalua ADL, asi que no alteran el contrato.
    salida["traza"]["query"] = decision["query"]
    salida["puntos"] = res.get("puntos", [])
    salida["grafico"] = res.get("grafico")
    salida["evidencia"] = res.get("evidencia", [])

    doc_ids = contrato.doc_ids_de(res["evaluacion"])
    _cerrar(salida, session_id, c.pregunta, rama, decision["fenomenos"], doc_ids,
            salida["metadata"]["estado"])
    return salida


def _cerrar(salida: dict, session_id: str, pregunta: str, rama: str,
            fenomenos: list[int], doc_ids: list[str], estado: str) -> None:
    """Memoria (escritura) + log JSONL: el cierre comun de todas las ramas."""
    md = salida["metadata"]
    memoria.escribir(session_id, pregunta, rama, fenomenos, doc_ids, estado)
    memoria.registrar({
        "ts": time.time(),
        "session_id": session_id,
        "pregunta": pregunta,
        "rama": rama,
        "fenomenos": fenomenos,
        "agentes_invocados": md["agentes_invocados"],
        "num_interacciones": md["num_interacciones"],
        "tokens": md["tokens"],
        "tokens_por_agente": md["tokens_por_agente"],
        "latencia_ms": md["latencia_ms"],
        "estado": estado,
        "doc_ids": doc_ids,
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", summary="Contrato oficial de evaluacion (seccion 2.4)")
def chat(c: Consulta) -> dict:
    """Responde una consulta con los bloques `respuesta`, `evaluacion` y `metadata`."""
    return resolver(c)


@app.post("/consulta", summary="Igual que /chat, usado por el frontend y el dashboard")
def consulta(c: Consulta) -> dict:
    """Alias de /chat. Existe para no romper el frontend del Reto 2, que ya lo consume."""
    return resolver(c)


@app.get("/agent-card", summary="Ficha del sistema multi-agente (seccion 2.3)")
def agent_card() -> JSONResponse:
    """Devuelve agent_card.json, la ficha estructural del equipo."""
    if not FICHA.exists():
        return JSONResponse({"error": "agent_card.json no encontrado"}, status_code=404)
    return JSONResponse(json.loads(FICHA.read_text(encoding="utf-8")))


@app.get("/salud", summary="Healthcheck del contenedor (Anexo A.4)")
def salud() -> dict:
    """Confirma que la base de conocimiento esta cargada y el servicio responde."""
    return {"ok": True, "equipo": config.EQUIPO, "version": app.version,
            "corpus": herramientas.estado(), "modelos": config.MODELO_POR_AGENTE}


@app.get("/", include_in_schema=False)
def inicio():
    """Sirve el frontend de chat cuando el contenedor tambien aloja el frontagent."""
    indice = RAIZ / "frontend" / "index.html"
    if indice.exists():
        return FileResponse(indice)
    return JSONResponse({"servicio": "CODEFEST AD ASTRA 2026 - Equipo Azor IV",
                         "endpoints": ["/chat", "/agent-card", "/salud", "/docs"]})
