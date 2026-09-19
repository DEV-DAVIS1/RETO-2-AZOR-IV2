"""
probar_comparador.py - Banco de prueba del COMPARADOR.
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Verifica lo que distingue a este agente: que la comparacion se apoye en evidencia
PAREADA y VALIDADA, no en un puente narrativo inventado por el modelo.

Una comparacion se da por aprobada cuando, tras la validacion en codigo, queda al
menos un item con una cita real por cada fenomeno comparado. Los items descartados
tambien se reportan: son alucinaciones que el modelo intento colar y que nunca
llegaron a la respuesta del usuario.

Requiere gateway y base vectorial. Consume presupuesto real de la bolsa de 100 USD
(seccion 1.3), por lo que informa el gasto acumulado al terminar.

Uso (PowerShell):
    $env:LLM_API_KEY = "SU_API_KEY"
    $env:LLM_BASE_URL = "https://litellm.admin-adl.codefest2026.augusta.avaldigitallabs.com"
    python -m pruebas.probar_comparador
Resultado detallado: pruebas/resultados/resultados_comparador.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAIError

from app import config
from app.agentes.comparador import comparar, registrar_gasto
from app.contrato import Contador

SALIDA = Path(__file__).resolve().parent / "resultados" / "resultados_comparador.jsonl"

#: (pregunta original, fenomenos, query) - exactamente lo que entrega el orquestador.
PREGUNTAS = [
    ("¿En qué se parecen el uso de IA militar y la seguridad espacial?",
     [1, 2], "riesgos seguridad regulación tecnología militar espacio"),
    ("Compara las dinámicas territoriales de América Latina con la militarización del espacio",
     [2, 3], "control territorial conflicto soberanía militarización"),
    ("¿Qué relación hay entre la IA en defensa y el control territorial en Latinoamérica?",
     [1, 3], "inteligencia artificial defensa vigilancia control territorial grupos armados"),
]


def main() -> int:
    if not os.getenv("LLM_API_KEY"):
        print("[ERROR] Falta LLM_API_KEY (API key del gateway de ADL).")
        return 1

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    aprobadas = 0
    total_usd = 0.0

    with SALIDA.open("w", encoding="utf-8") as salida:
        for n, (pregunta, fenomenos, query) in enumerate(PREGUNTAS, 1):
            contador = Contador()
            try:
                res = comparar(pregunta, fenomenos, query, contador)
            except OpenAIError as e:
                print(f"[ERROR gateway] {e!r}"[:600])
                return 1

            diag = res["diagnostico"]
            tokens = contador.totales()
            total_usd = registrar_gasto(diag["costo_usd"], tokens["total"])
            validos = diag["coincidencias_validas"] + diag["diferencias_validas"]
            ok = res["estado"] == "ok" and validos >= 1
            aprobadas += ok

            salida.write(json.dumps(
                {"pregunta": pregunta, "fenomenos": fenomenos, "respuesta": res["respuesta"],
                 "evaluacion": res["evaluacion"], "tokens": tokens,
                 "tokens_por_agente": contador.por_agente(), "diagnostico": diag},
                ensure_ascii=False) + "\n")

            print("=" * 78)
            print(f"[{n}] {pregunta}")
            print("=" * 78)
            print(res["respuesta"])
            print("-" * 78)
            costo = f"${diag['costo_usd']:.5f}" if diag["costo_usd"] is not None else "no reportado"
            print(f"Tokens: entrada {tokens['input']}, salida {tokens['output']} | "
                  f"{diag['latencia_ms']} ms | costo {costo}")
            print(f"Estado: {res['estado']} | items validos: "
                  f"{diag['coincidencias_validas']} coinc. + {diag['diferencias_validas']} dif. | "
                  f"descartados por validacion: {diag['items_descartados']} "
                  f"-> {'OK' if ok else 'REVISAR'}")

    print("\n" + "=" * 78)
    print(f"MODELO: {config.MODELO_COMPARADOR}")
    print(f"  {aprobadas}/{len(PREGUNTAS)} comparaciones con evidencia pareada valida")
    print(f"  Gasto acumulado segun el gateway: ${total_usd:.4f} USD de $100")
    print(f"  VEREDICTO: {'APROBADO' if aprobadas >= 2 else 'NO APROBADO'}")
    print(f"  Detalle en: {SALIDA}")
    return 0 if aprobadas >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
