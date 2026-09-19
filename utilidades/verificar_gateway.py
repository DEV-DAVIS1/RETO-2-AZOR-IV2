"""
verificar_gateway.py - Comprobacion previa del gateway de modelos de ADL.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Conviene ejecutarlo antes de abrir la ventana de evaluacion y despues de cada
cambio de modelo en app/config.py. Responde tres preguntas:

  1. Que modelos expone realmente el gateway (sus identificadores exactos).
  2. Si los cuatro modelos que el sistema tiene configurados responden.
  3. Cuanto cuesta una llamada minima a cada uno, segun la cabecera
     x-litellm-response-cost.

Un identificador equivocado en config.py no falla al arrancar: falla en la
primera pregunta del evaluador. Esta comprobacion adelanta ese fallo.

Uso (PowerShell):
    $env:LLM_API_KEY  = "SU_API_KEY"
    $env:LLM_BASE_URL = "https://litellm.admin-adl.codefest2026.augusta.avaldigitallabs.com"
    python -m utilidades.verificar_gateway
"""
from __future__ import annotations

import os
import sys
import time

from openai import OpenAI, OpenAIError

from app import config


def listar_modelos(cliente: OpenAI) -> list[str]:
    """Identificadores que el gateway declara disponibles."""
    try:
        return sorted(m.id for m in cliente.models.list().data)
    except OpenAIError as e:
        print(f"  No se pudo listar el catalogo: {type(e).__name__}: {e}"[:300])
        return []


def probar_modelo(cliente: OpenAI, modelo: str) -> tuple[bool, str]:
    """Llamada minima para confirmar que el modelo responde y cuanto cuesta."""
    t0 = time.perf_counter()
    try:
        crudo = cliente.chat.completions.with_raw_response.create(
            model=modelo,
            messages=[{"role": "user", "content": "Responde unicamente: ok"}],
            temperature=0, max_tokens=5)
        r = crudo.parse()
        ms = int((time.perf_counter() - t0) * 1000)
        costo = crudo.headers.get("x-litellm-response-cost")
        detalle = (f"{r.usage.total_tokens} tokens | {ms} ms | "
                   f"costo {'$' + costo if costo else 'no reportado'}")
        return True, detalle
    except OpenAIError as e:
        return False, f"{type(e).__name__}: {e}"[:200]


def main() -> int:
    if not os.getenv("LLM_API_KEY"):
        print("[ERROR] Falta LLM_API_KEY.")
        return 1

    print(f"Gateway: {config.url_v1()}\n")
    cliente = OpenAI(base_url=config.url_v1(), api_key=config.LLM_API_KEY,
                     timeout=60, max_retries=0)

    print("CATALOGO DEL GATEWAY")
    disponibles = listar_modelos(cliente)
    for modelo in disponibles:
        print(f"  {modelo}")
    if not disponibles:
        print("  (vacio: el gateway puede no exponer /models; se prueba igual)")

    print("\nMODELOS CONFIGURADOS EN app/config.py")
    fallos = []
    for agente, modelo in config.MODELO_POR_AGENTE.items():
        en_catalogo = (not disponibles) or (modelo in disponibles)
        ok, detalle = probar_modelo(cliente, modelo)
        marca = "OK  " if ok else "FALLA"
        aviso = "" if en_catalogo else "  <- no aparece en el catalogo"
        print(f"  {marca} {agente:<13} {modelo:<32} {detalle}{aviso}")
        if not ok:
            fallos.append((agente, modelo))

    if fallos:
        print("\n  Modelos que no respondieron:")
        for agente, modelo in fallos:
            print(f"   - {agente}: {modelo}")
        print("\n  Corrige el identificador en app/config.py (o en la variable de entorno")
        print("  correspondiente) y actualiza agent_card.json para que coincidan.")
        return 1

    print("\n  Los cuatro modelos responden. El sistema puede desplegarse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
