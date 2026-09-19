"""
inspeccionar.py — Diagnóstico de la base vectorial de la Etapa 1 (CODEFEST AD ASTRA 2026)

Detecta automáticamente el índice FAISS y la metadata, y reporta lo necesario para
configurar rag_local.py: tipo de índice, n.º de vectores, dimensión, columnas,
cobertura de 'localizacion' y pistas del modelo de embeddings.

Uso (desde la carpeta del proyecto):
    python inspeccionar.py                      # busca en ./base_vectorial
    python inspeccionar.py --carpeta RUTA       # otra carpeta
"""
import argparse
import json
import re
from pathlib import Path

import faiss
import pandas as pd

EXT_INDICE = {".faiss", ".index", ".idx"}
EXT_META = {".parquet", ".jsonl", ".json", ".pkl", ".pickle", ".csv"}
COLUMNAS_OBLIGATORIAS = ["doc_id", "chunk_id", "fuente", "formato", "fenomeno",
                         "posicion", "num_tokens", "texto"]
DIMENSIONES_CONOCIDAS = {
    384: "all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2 / multilingual-e5-small / bge-small",
    512: "distiluse-base-multilingual-cased",
    768: "paraphrase-multilingual-mpnet-base-v2 / multilingual-e5-base / LaBSE / bge-base",
    1024: "multilingual-e5-large / bge-m3 / bge-large",
    1536: "OpenAI text-embedding-3-small / ada-002",
    3072: "OpenAI text-embedding-3-large",
}


def titulo(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def cargar_metadata(ruta: Path) -> pd.DataFrame:
    s = ruta.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(ruta)
    if s == ".jsonl":
        return pd.read_json(ruta, lines=True)
    if s == ".json":
        try:
            return pd.read_json(ruta)
        except ValueError:
            return pd.read_json(ruta, lines=True)
    if s in {".pkl", ".pickle"}:
        obj = pd.read_pickle(ruta)
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    if s == ".csv":
        return pd.read_csv(ruta)
    raise ValueError(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="base_vectorial")
    carpeta = Path(ap.parse_args().carpeta)

    # 1. Archivos -----------------------------------------------------------
    titulo(f"1. ARCHIVOS EN {carpeta.resolve()}")
    if not carpeta.exists():
        print(f"[ERROR] No existe la carpeta '{carpeta}'. Usa --carpeta RUTA")
        return
    archivos = [p for p in carpeta.rglob("*") if p.is_file()]
    for p in archivos:
        print(f"  {p.relative_to(carpeta)}  ({p.stat().st_size / 1e6:,.1f} MB)")

    indices = [p for p in archivos if p.suffix.lower() in EXT_INDICE]
    metas = [p for p in archivos if p.suffix.lower() in EXT_META]

    # 2. Índice FAISS -------------------------------------------------------
    titulo("2. ÍNDICE FAISS")
    todos = []  # (ruta, indice)
    for p in indices:
        try:
            idx = faiss.read_index(str(p))
            todos.append((p, idx))
            print(f"\n  Archivo    : {p}")
            print(f"  Tipo       : {type(idx).__name__}")
            print(f"  Vectores   : {idx.ntotal:,}")
            print(f"  Dimensión  : {idx.d}")
            metrica = "Producto interno (coseno si normalizado) -> NORMALIZAR=True" \
                if idx.metric_type == faiss.METRIC_INNER_PRODUCT \
                else "L2 (euclidiana) -> NORMALIZAR=False (salvo que se normalizara al indexar)"
            print(f"  Métrica    : {metrica}")
            print(f"  Posibles modelos por dimensión: "
                  f"{DIMENSIONES_CONOCIDAS.get(idx.d, 'dimensión no típica — revisar código Etapa 1')}")
        except Exception as e:  # noqa: BLE001
            print(f"  [AVISO] {p} no se pudo leer como FAISS: {e}")
    if not todos:
        print("  [ERROR] No se encontró un índice FAISS legible.")
    ruta_indice, indice = todos[0] if todos else (None, None)

    # 3. Metadata -----------------------------------------------------------
    titulo("3. METADATA")
    df = None
    ruta_meta = None
    for p in metas:
        try:
            cand = cargar_metadata(p)
            if len(cand) > 0:
                df = cand
                ruta_meta = p
                print(f"  Archivo    : {p}")
                print(f"  Filas      : {len(df):,}")
                print(f"  Columnas   : {df.columns.tolist()}")
                break
        except Exception as e:  # noqa: BLE001
            print(f"  [AVISO] {p} no se pudo leer: {e}")
    if df is None:
        print("  [ERROR] No se encontró metadata legible.")
        return

    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    print(f"  Obligatorias faltantes: {faltan if faltan else 'ninguna'}")
    print("\n  Primera fila:")
    fila = df.iloc[0].to_dict()
    for k, v in fila.items():
        print(f"    {k:<15}: {str(v)[:100]}")

    if "fenomeno" in df.columns:
        print("\n  Fragmentos por fenómeno:")
        print(df["fenomeno"].value_counts().sort_index().to_string())

    # 4. Localización -------------------------------------------------------
    titulo("4. CAMPO DE LOCALIZACIÓN")
    col_loc = [c for c in df.columns if re.search(r"loc|coord|lat|lon|geo", c, re.I)]
    if not col_loc:
        print("  No se encontró columna de localización.")
    for c in col_loc:
        cobertura = df[c].notna().mean() * 100
        print(f"  Columna '{c}': cobertura {cobertura:.1f} %")
        print(f"  Ejemplos: {df[c].dropna().head(3).tolist()}")
        if "fenomeno" in df.columns:
            print("  Cobertura por fenómeno (%):")
            print((df.groupby('fenomeno')[c].apply(lambda s: s.notna().mean() * 100))
                  .round(1).to_string())

    # 5. Alineación índice ↔️ metadata --------------------------------------
    titulo("5. ALINEACIÓN ÍNDICE ↔️ METADATA")
    if indice is not None:
        if indice.ntotal == len(df):
            print(f"  OK: {indice.ntotal:,} vectores = {len(df):,} filas")
        else:
            print(f"  [ERROR] {indice.ntotal:,} vectores ≠ {len(df):,} filas "
                  f"→ el orden fila i ↔️ vector i NO es confiable")

    # 6. Pistas del modelo de embeddings -----------------------------------
    titulo("6. MODELO DE EMBEDDINGS (búsqueda en el código del proyecto)")
    patron = re.compile(r"(SentenceTransformer\(|model_name|embedding|encode\()", re.I)
    hallazgos = 0
    for p in Path(".").rglob("*"):
        if hallazgos >= 25:
            break
        if p.suffix not in {".py", ".ipynb", ".json", ".yaml", ".yml", ".txt", ".md"}:
            continue
        if ".venv" in p.parts or p.stat().st_size > 5e6:
            continue
        try:
            texto = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if p.suffix == ".ipynb":
            try:
                texto = "\n".join("".join(c.get("source", []))
                                  for c in json.loads(texto).get("cells", []))
            except json.JSONDecodeError:
                pass
        for n, linea in enumerate(texto.splitlines(), 1):
            if patron.search(linea):
                print(f"  {p}:{n}: {linea.strip()[:120]}")
                hallazgos += 1
                if hallazgos >= 25:
                    break
    if hallazgos == 0:
        print("  Sin coincidencias: revisar generador.py / informe técnico de la Etapa 1.")

    titulo("RESUMEN PARA rag_local.py")
    if indice is not None:
        print(f"  RUTA_INDICE       = {ruta_indice}")
        print(f"  NORMALIZAR        = {indice.metric_type == faiss.METRIC_INNER_PRODUCT}")
        print(f"  Dimensión a igualar con MODELO_EMBEDDINGS: {indice.d}")
    print(f"  RUTA_METADATA     = {ruta_meta}")


if __name__ == "__main__":
    main()