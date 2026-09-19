"""
verificar_modelos.py — Identifica con certeza qué modelo de embeddings construyó cada índice.

Los 3 índices tienen dimensión 1024, así que la dimensión NO basta para distinguir
bge-m3 / multilingual-e5-large / multilingual-e5-large-instruct. Este script:
  1. Toma 3 fragmentos de la metadata.
  2. Recupera su vector guardado en cada índice (index.reconstruct).
  3. Recalcula el vector con cada modelo candidato y cada prefijo de documento.
  4. Compara (coseno). La combinación correcta da ~0.99 o más.

Uso:  python verificar_modelos.py
Nota: la primera vez descarga los 3 modelos (~2 GB cada uno).
"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

RUTA_METADATA = "base_vectorial/Metadata_AZOR_IV.jsonl"
INDICES = {
    "encoder_bge": "base_vectorial/encoder_bge/index.faiss",
    "encoder_e5": "base_vectorial/encoder_e5/index.faiss",
    "encoder_e5_instruct": "base_vectorial/encoder_e5_instruct/index.faiss",
}
CANDIDATOS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
]
PREFIJOS_DOC = ["", "passage: "]   # cómo pudo haberse codificado el fragmento
FILAS_PRUEBA = [0, 50000, 120000]

# Modo rápido: solo verifica bge-m3 contra encoder_bge (1 descarga de ~2 GB)
import sys
if "--rapido" in sys.argv:
    INDICES = {"encoder_bge": INDICES["encoder_bge"]}
    CANDIDATOS = ["BAAI/bge-m3"]

# Leer solo las filas necesarias del JSONL (evita cargar 250 MB)
textos = {}
with open(RUTA_METADATA, encoding="utf-8") as f:
    for n, linea in enumerate(f):
        if n in FILAS_PRUEBA:
            textos[n] = json.loads(linea)["texto"]
        if n > max(FILAS_PRUEBA):
            break

indices = {nombre: faiss.read_index(ruta) for nombre, ruta in INDICES.items()}
guardados = {nombre: np.stack([idx.reconstruct(i) for i in FILAS_PRUEBA])
             for nombre, idx in indices.items()}

print(f"{'índice':<22}{'modelo candidato':<42}{'prefijo':<12}{'coseno medio':>12}")
print("-" * 88)
mejor = {}
for modelo in CANDIDATOS:
    enc = SentenceTransformer(modelo)
    for prefijo in PREFIJOS_DOC:
        nuevos = enc.encode([prefijo + textos[i] for i in FILAS_PRUEBA],
                            normalize_embeddings=True)
        for nombre, g in guardados.items():
            gn = g / np.linalg.norm(g, axis=1, keepdims=True)
            cos = float(np.mean(np.sum(gn * nuevos, axis=1)))
            print(f"{nombre:<22}{modelo:<42}{repr(prefijo):<12}{cos:>12.4f}")
            if cos > mejor.get(nombre, (None, None, -1))[2]:
                mejor[nombre] = (modelo, prefijo, cos)
    del enc

print("\nRESULTADO (usar en rag_local.py):")
for nombre, (modelo, prefijo, cos) in mejor.items():
    veredicto = "CONFIRMADO" if cos >= 0.98 else "DUDOSO: revisar código de la Etapa 1"
    print(f"  {nombre:<22} → {modelo}  | prefijo de documento {prefijo!r} | coseno {cos:.4f}  [{veredicto}]")
