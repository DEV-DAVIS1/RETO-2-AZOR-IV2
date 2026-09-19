"""
llm.py - Cliente unico hacia el gateway de modelos de ADL.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Los ocho modelos del reto se consumen via API por el gateway LiteLLM sobre
Amazon Bedrock. Un solo cliente compartido reutiliza la conexion HTTP entre
agentes, que de otro modo pagarian el handshake TLS en cada llamada.

`extra_body` resuelve una diferencia real entre entornos: `reasoning_effort`
recorta los tokens de razonamiento de gpt-oss en Ollama local, pero el gateway
de Bedrock puede rechazar el parametro. Se envia solo donde aplica.
"""
from __future__ import annotations

import functools
import logging

from openai import OpenAI

from . import config

log = logging.getLogger("llm")


@functools.lru_cache(maxsize=1)
def cliente() -> OpenAI:
    """Cliente compartido del proceso, creado la primera vez que se necesita."""
    return OpenAI(base_url=config.url_v1(), api_key=config.LLM_API_KEY,
                  timeout=config.TIMEOUT_LLM_S, max_retries=1)


def extra_body(modelo: str) -> dict:
    """Parametros que solo acepta el runtime local."""
    return {"reasoning_effort": "low"} if (config.ES_LOCAL and "gpt-oss" in modelo.lower()) else {}


def uso(respuesta) -> tuple[int, int]:
    """(tokens_entrada, tokens_salida) tolerando respuestas sin bloque `usage`."""
    u = getattr(respuesta, "usage", None)
    return (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
