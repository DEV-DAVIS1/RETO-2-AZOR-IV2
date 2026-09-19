"""
probar_visualizador.py — Banco de prueba del VISUALIZADOR (CODEFEST AD ASTRA 2026)

Parte A (sin LLM, 0 tokens): los once componentes + la heuristica de respaldo.
Parte B (con LLM): set etiquetado. Mide % tool_call emitido, % componente
correcto, % fenomeno correcto, tokens y latencia.

Uso:
    python -m pruebas.probar_visualizador                   # A + B con gpt-oss:20b en Ollama
    python -m pruebas.probar_visualizador --sin-llm         # solo A
    python -m pruebas.probar_visualizador --modelo gemma3:27b
Detalle: pruebas/resultados/resultados_visualizador.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from app import config
from app.agentes.visualizador import (generar_visualizacion, get_motor,
                                      inferir_componente, visualizar)

SALIDA = Path(__file__).resolve().parent / "resultados" / "resultados_visualizador.jsonl"

# (pregunta, componente esperado, fenomeno esperado | None)
PRUEBAS = [
    ("Muéstrame en el mapa dónde se concentran los grupos armados", "mapa_puntos", 3),
    ("Muéstrame un mapa de los lugares mencionados sobre conflicto en Colombia", "mapa_puntos", 3),
    ("¿Qué países concentran más menciones sobre IA militar?", "mapa_coropletico", 1),
    ("Compara en un mapa los países de la región con más menciones de economías ilegales",
     "mapa_coropletico", 3),
    ("¿Cómo evolucionó el tema de la basura espacial entre 2018 y 2025?", "linea_tiempo", 2),
    ("Grafica cuántos documentos hay por año sobre basura espacial", "linea_tiempo", 2),
    ("Muéstrame la tendencia de las pruebas antisatélite en los últimos años", "linea_tiempo", 2),
    ("Cruza los países con los tres fenómenos", "matriz_calor", None),
    ("Haz un mapa de calor de países por fenómeno", "matriz_calor", None),
    ("¿Qué actores se relacionan entre sí en el conflicto armado?", "red_coocurrencia", 3),
    ("Visualiza las conexiones entre países en seguridad espacial", "red_coocurrencia", 2),
    ("Muéstrame las fuentes en que te basas", "panel_evidencia", None),
    ("¿Cuántos documentos hay por observatorio en seguridad espacial?", "barras", 2),
    ("Proporción de idiomas por fenómeno", "composicion", None),
    ("Histograma del tamaño de los fragmentos", "histograma", None),
    ("¿Qué países merecen atención con más urgencia en narcotráfico?", "cuadrante", 3),
    ("Muéstrame las zonas calientes de minería ilegal", "mapa_calor", 3),
]


def _n(res):
    d = res["datos"]
    return len(d) if isinstance(d, list) else len(d.get("aristas", d.get("puntos", [])))


def parte_a():
    print("=" * 78 + "\nPARTE A — componentes sin LLM\n" + "=" * 78)
    t0 = time.time()
    m = get_motor()
    print(f"metadata: {len(m.chunks):,} chunks en {time.time() - t0:.1f}s | "
          f"geo={m.tiene_geo} fechas={m.tiene_fechas}\n")
    casos = [("donde operan los grupos armados", "mapa_puntos", 3),
             ("que paises concentran mas menciones", "mapa_coropletico", 3),
             ("evolucion entre 2018 y 2025", "linea_tiempo", 2),
             ("cruza paises y fenomenos", "matriz_calor", None),
             ("que actores se relacionan", "red_coocurrencia", 1),
             ("en que te basas", "panel_evidencia", 3),
             ("zonas calientes", "mapa_calor", 3),
             ("documentos por observatorio", "barras", None),
             ("idiomas por fenomeno", "composicion", None),
             ("tamano de los fragmentos", "histograma", None),
             ("que paises priorizar", "cuadrante", 3)]
    for instr, comp, fen in casos:
        t = time.perf_counter()
        r = generar_visualizacion(instr, comp, fen)
        ms = (time.perf_counter() - t) * 1000
        print(f"  {r['componente']:18} n={_n(r):4}  docs={r['trazabilidad']['documentos']:4}"
              f"  {ms:7.0f} ms  | {r['titulo']}")

    ok = sum(inferir_componente(p) == c for p, c, _ in PRUEBAS)
    print(f"\n  Heuristica de respaldo: {ok}/{len(PRUEBAS)} componentes correctos")
    for p, c, _ in PRUEBAS:
        h = inferir_componente(p)
        if h != c:
            print(f"   - esperado {c:<18} heuristica {h:<18} | {p}")


def parte_b(modelo):
    print("\n" + "=" * 78 + f"\nPARTE B — agente con LLM ({modelo})\n" + "=" * 78)
    ok_tool = ok_comp = ok_fen = tin = tout = lat = 0
    errores = []
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as f:
        for k, (preg, comp_esp, fen_esp) in enumerate(PRUEBAS, 1):
            # Se simula la salida del orquestador: solo pasa fenomeno si es unico
            out = visualizar(preg, [fen_esp] if fen_esp else None, modelo=modelo)
            meta, args = out["metadata"], out["evaluacion"]["tools_called"][0]["input_parameters"]
            comp = out["visualizacion"]["componente"]
            tool = meta["estado"] == "ok" and not out["visualizacion"].get(
                "componente_inferido_por_heuristica")
            ok_tool += tool
            ok_comp += comp == comp_esp
            ok_fen += args.get("fenomeno") == fen_esp
            tin += meta["tokens"]["input"]
            tout += meta["tokens"]["output"]
            lat += meta["latencia_ms"]
            print(f"[{k:02d}] {'OK ' if comp == comp_esp else 'MAL'} {comp:<18} "
                  f"esp={comp_esp:<18} fen={args.get('fenomeno')} tool={'si' if tool else 'NO'} "
                  f"| {meta['tokens']['total']:>5} tok | {meta['latencia_ms']:>6} ms")
            if comp != comp_esp:
                errores.append((preg, comp_esp, comp))
            f.write(json.dumps({"pregunta": preg, "esperado": comp_esp, **out},
                               ensure_ascii=False, default=str) + "\n")

    n = len(PRUEBAS)
    print("\n" + "=" * 70)
    print(f"  Tool call del LLM  : {ok_tool}/{n} = {ok_tool / n:.0%}")
    print(f"  Componente correcto: {ok_comp}/{n} = {ok_comp / n:.0%}   (meta >= 90 %)")
    print(f"  Fenomeno correcto  : {ok_fen}/{n} = {ok_fen / n:.0%}")
    print(f"  Tokens promedio    : entrada {tin // n}, salida {tout // n}")
    print(f"  Latencia promedio  : {lat // n} ms (local; NO representa Bedrock)")
    for p, e, o in errores:
        print(f"   - esperado {e:<18} obtenido {o:<18} | {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=config.MODELO_VISUALIZADOR)
    ap.add_argument("--sin-llm", action="store_true")
    a = ap.parse_args()
    parte_a()
    if not a.sin_llm:
        parte_b(a.modelo)
