"""
config.py - Configuracion central del sistema multi-agente.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Toda la configuracion se inyecta por variables de entorno desde Coolify
(Anexo A.6 de la especificacion): ninguna credencial vive en el codigo.

En produccion los cuatro agentes consumen los modelos de ADL a traves del
gateway LiteLLM (Amazon Bedrock). En desarrollo local basta apuntar
LLM_BASE_URL a Ollama.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Identidad del equipo
# ---------------------------------------------------------------------------
EQUIPO = os.getenv("EQUIPO", "azoriv")
DOMINIO_BASE = os.getenv("DOMINIO_BASE", "codefest2026.augusta.avaldigitallabs.com")
ENDPOINT_PUBLICO = os.getenv("ENDPOINT_PUBLICO", f"https://agent.{EQUIPO}.{DOMINIO_BASE}/chat")

# ---------------------------------------------------------------------------
# Gateway de modelos (Amazon Bedrock via LiteLLM de ADL)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

#: True cuando se trabaja contra Ollama local (habilita `reasoning_effort`,
#: que el gateway de Bedrock puede rechazar).
ES_LOCAL = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL


def url_v1(base: str = LLM_BASE_URL) -> str:
    """Normaliza la URL del gateway: LiteLLM se declara con o sin sufijo /v1."""
    base = base.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


# ---------------------------------------------------------------------------
# Modelo por agente. Estos valores deben coincidir con agent_card.json:
# la ficha es el dato estructural con el que ADL calcula el costo (seccion 2.5).
# ---------------------------------------------------------------------------
MODELO_ORQUESTADOR = os.getenv("MODELO_ORQUESTADOR", "gpt-oss-20b")
MODELO_REDACTOR = os.getenv("MODELO_REDACTOR", "meta.llama3-3-70b-instruct")
MODELO_COMPARADOR = os.getenv("MODELO_COMPARADOR", "meta.llama3-3-70b-instruct")
MODELO_VISUALIZADOR = os.getenv("MODELO_VISUALIZADOR", "gpt-oss-20b")

MODELO_POR_AGENTE = {
    "orquestador": MODELO_ORQUESTADOR,
    "redactor": MODELO_REDACTOR,
    "comparador": MODELO_COMPARADOR,
    "visualizador": MODELO_VISUALIZADOR,
}

# ---------------------------------------------------------------------------
# Base de conocimiento de la Etapa 1
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
BASE_VECTORIAL = Path(os.getenv("BASE_VECTORIAL", RAIZ / "base_vectorial"))
RUTA_METADATA = os.getenv("METADATA_PATH", str(BASE_VECTORIAL / "Metadata_AZOR_IV.jsonl"))

#: Los tres indices FAISS de la Etapa 1. Se fusionan con Reciprocal Rank Fusion.
#: RAG_INDICES permite degradar a menos indices si hace falta recortar latencia.
_INDICES_DISPONIBLES = {
    "bge": {"ruta": str(BASE_VECTORIAL / "encoder_bge" / "index.faiss"),
            "modelo": "BAAI/bge-m3", "normalizar": True, "prefijo": ""},
    "e5": {"ruta": str(BASE_VECTORIAL / "encoder_e5" / "index.faiss"),
           "modelo": "intfloat/multilingual-e5-large", "normalizar": True,
           "prefijo": "query: "},
    "e5_instruct": {"ruta": str(BASE_VECTORIAL / "encoder_e5_instruct" / "index.faiss"),
                    "modelo": "intfloat/multilingual-e5-large-instruct", "normalizar": True,
                    "prefijo": "Instruct: Given a question, retrieve relevant passages that "
                               "answer the question\nQuery: "},
}
_SELECCION = [n.strip() for n in os.getenv("RAG_INDICES", "bge,e5,e5_instruct").split(",") if n.strip()]
INDICES = [dict(_INDICES_DISPONIBLES[n], nombre=n) for n in _SELECCION if n in _INDICES_DISPONIBLES]

TOP_K = int(os.getenv("TOP_K", "5"))              # fragmentos finales entregados al LLM
CANDIDATOS = int(os.getenv("CANDIDATOS", "30"))   # candidatos por indice antes de fusionar
RRF_K = int(os.getenv("RRF_K", "60"))             # constante estandar de RRF
FRAGMENTOS_POR_FENOMENO = int(os.getenv("FRAGMENTOS_POR_FENOMENO", "3"))

# La lista de fuentes viaja siempre en el campo `fuentes` y el frontend la dibuja
# bajo la respuesta. Anexarla ademas al texto esta apagado por defecto: `respuesta`
# es tambien `actual_output` (seccion 2.4), y titulos, fechas e ids que no estan en
# `retrieval_context` pueden restar en Faithfulness y Answer Relevancy (2.5.1).
FUENTES_EN_TEXTO = os.getenv("FUENTES_EN_TEXTO", "0").strip().lower() in ("1", "true", "si")


def _bandera(nombre: str, defecto: str) -> bool:
    return os.getenv(nombre, defecto).strip().lower() in ("1", "true", "si")


# Metadata enriquecida (geo, fechas) en la recuperacion. Ver app/lugares.py.
#: Si la pregunta nombra un pais o un anio, buscar_corpus antepone los fragmentos
#: que los contienen. Es una preferencia: nunca deja al redactor sin evidencia.
FILTRO_GEO = _bandera("FILTRO_GEO", "1")
#: Candidatos por indice cuando hay filtro: hacen falta mas que CANDIDATOS para que
#: el filtro tenga de donde elegir. FAISS recorre el indice completo igual.
CANDIDATOS_FILTRO = int(os.getenv("CANDIDATOS_FILTRO", "100"))
#: Si TODAS las citas del redactor son inventadas, la respuesta no tiene respaldo
#: verificable y se sustituye por la salida fija de "sin informacion".
BLOQUEAR_CITAS_INVENTADAS = _bandera("BLOQUEAR_CITAS_INVENTADAS", "1")
#: Las respuestas de texto traen `puntos`: los lugares que mencionan, sacados de
#: los fragmentos que citan. Es lo que dibuja el mapa del dashboard.
PUNTOS_EN_RESPUESTAS = _bandera("PUNTOS_EN_RESPUESTAS", "1")

# ---------------------------------------------------------------------------
# Memoria conversacional y trazabilidad
# ---------------------------------------------------------------------------
DIR_ESTADO = Path(os.getenv("DIR_ESTADO", RAIZ / "estado"))
RUTA_LOG = Path(os.getenv("RUTA_LOG", DIR_ESTADO / "interacciones.jsonl"))
MEMORIA_TURNOS = int(os.getenv("MEMORIA_TURNOS", "6"))     # turnos que se conservan por sesion
MEMORIA_SESIONES = int(os.getenv("MEMORIA_SESIONES", "500"))  # sesiones vivas en RAM
MEMORIA_TTL_S = int(os.getenv("MEMORIA_TTL_S", "3600"))    # caducidad de una sesion inactiva

# ---------------------------------------------------------------------------
# Servicio HTTP
# ---------------------------------------------------------------------------
PUERTO = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
TIMEOUT_LLM_S = float(os.getenv("TIMEOUT_LLM_S", "90"))

NOMBRE_FENOMENO = {
    1: "IA en entornos militares",
    2: "seguridad espacial y orbita baja terrestre",
    3: "dinamicas territoriales en America Latina",
}

MENSAJE_FUERA_DOMINIO = (
    "Esa consulta esta fuera del alcance de este asistente. Puedo ayudarte con analisis "
    "documental de tres fenomenos: inteligencia artificial en entornos militares, seguridad "
    "espacial y orbita baja terrestre, y dinamicas territoriales en America Latina. "
    "Reformula tu pregunta sobre alguno de ellos y con gusto la respondo."
)
