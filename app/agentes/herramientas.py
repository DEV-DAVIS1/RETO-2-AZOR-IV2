"""
herramientas.py - Tools declaradas en la ficha del agente (agent_card.json).
CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV

Son las dos cajas naranjas del diagrama de arquitectura:

  buscar_corpus    recuperacion semantica sobre la base vectorial de la Etapa 1.
                   La usan el orquestador (indirectamente), el redactor y el comparador.
  consultar_datos  agregacion sobre la metadata real, sin generacion. La usa el
                   visualizador para poblar los componentes del tablero.

La base de conocimiento se carga UNA sola vez al importar el modulo: tres indices
FAISS y sus tres modelos de embeddings. Es la operacion cara del arranque
(varios minutos en frio), por eso el contenedor declara un `start-period` amplio
en su healthcheck.

Fusion de los tres indices con Reciprocal Rank Fusion:

    puntaje(fragmento) = SUM_indices  1 / (RRF_K + posicion_en_ese_indice)

RRF combina rankings de modelos de embeddings distintos sin necesidad de
calibrar sus escalas de similitud, que no son comparables entre si.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from .. import config

log = logging.getLogger("herramientas")


# ---------------------------------------------------------------------------
# Carga de la base de conocimiento (una sola vez por proceso)
# ---------------------------------------------------------------------------
def _cargar_metadata(ruta: str) -> pd.DataFrame:
    if ruta.endswith(".parquet"):
        return pd.read_parquet(ruta)
    if ruta.endswith(".jsonl"):
        return pd.read_json(ruta, lines=True)
    if ruta.endswith(".pkl"):
        return pd.read_pickle(ruta)
    raise ValueError(f"Formato de metadata no soportado: {ruta}")


def _verificar_archivo(ruta: str) -> None:
    """Comprueba que el archivo existe y no es un puntero de Git LFS.

    Es el fallo mas probable del despliegue: si el entorno que clona el
    repositorio no tiene git-lfs instalado, en lugar del indice de 590 MB queda
    un archivo de texto de ~130 bytes. FAISS fallaria mas adelante con un error
    ilegible; conviene detectarlo aqui con un mensaje que diga que hacer.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        raise RuntimeError(
            f"No se encuentra {ruta}. Revisa BASE_VECTORIAL / METADATA_PATH, o que el "
            f"clon del repositorio haya traido la base vectorial.")
    if archivo.stat().st_size < 1_000_000:
        cabecera = archivo.open("rb").read(60)
        if cabecera.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{ruta} es un puntero de Git LFS, no el archivo real. Ejecuta "
                f"'git lfs install && git lfs pull' en el entorno de despliegue.")


_t0 = time.perf_counter()
_verificar_archivo(config.RUTA_METADATA)
metadata = _cargar_metadata(config.RUTA_METADATA)

INDICES = [dict(cfg) for cfg in config.INDICES]
for _cfg in INDICES:
    _verificar_archivo(_cfg["ruta"])
    _cfg["indice"] = faiss.read_index(_cfg["ruta"])
    _cfg["embedder"] = SentenceTransformer(_cfg["modelo"])
    _dim = _cfg["embedder"].get_sentence_embedding_dimension()
    # Verificaciones criticas: un indice construido con otro modelo devolveria
    # vecinos sin sentido en silencio, sin lanzar ningun error.
    if _cfg["indice"].ntotal != len(metadata):
        raise RuntimeError(f"{_cfg['ruta']}: {_cfg['indice'].ntotal} vectores != "
                           f"{len(metadata)} filas de metadata")
    if _cfg["indice"].d != _dim:
        raise RuntimeError(f"{_cfg['ruta']}: dimension {_cfg['indice'].d} != {_dim} de "
                           f"{_cfg['modelo']}; ese modelo NO construyo este indice")

#: Columna de fenomeno como array, para filtrar sin recorrer el DataFrame.
FENOMENO_POR_FILA = metadata["fenomeno"].astype(int).to_numpy()
log.info("Base de conocimiento lista: %d fragmentos, %d indices, %.1f s",
         len(metadata), len(INDICES), time.perf_counter() - _t0)


# ---------------------------------------------------------------------------
# Tool: buscar_corpus
# ---------------------------------------------------------------------------
#: Campos de la metadata con los que se arma la cita que ve el usuario. No todos
#: los documentos del corpus los traen completos, por eso se leen con tolerancia.
CAMPOS_DE_FUENTE = ("fuente", "titulo", "formato", "posicion",
                    "fecha_publicacion", "observatorio", "doc_id_oficial")


def _valor(fila, campo: str):
    """Valor de un campo opcional de la metadata, o None si falta o es nulo.

    Devuelve tipos nativos de Python: pandas entrega `numpy.int64` en columnas
    como `posicion`, y FastAPI no sabe serializarlo (la respuesta caeria en 500).
    """
    if campo not in fila.index:
        return None
    valor = fila[campo]
    if pd.isna(valor):
        return None
    return valor.item() if hasattr(valor, "item") else valor


def buscar_corpus(query: str, fenomeno: Optional[int] = None,
                  k: int = config.TOP_K) -> list[dict]:
    """Recupera los `k` fragmentos mas relevantes del corpus.

    `fenomeno` restringe la busqueda a uno de los tres fenomenos del reto; es el
    filtro que permite al comparador traer evidencia equilibrada de cada lado.
    """
    puntajes: dict[int, float] = {}
    votos: dict[int, list[str]] = {}
    for cfg in INDICES:
        vec = cfg["embedder"].encode([cfg["prefijo"] + query],
                                     normalize_embeddings=cfg["normalizar"]).astype("float32")
        _, ids = cfg["indice"].search(vec, config.CANDIDATOS)
        posicion = 0
        for i in ids[0]:
            if i == -1:
                continue
            if fenomeno and int(FENOMENO_POR_FILA[i]) != int(fenomeno):
                continue
            posicion += 1
            puntajes[i] = puntajes.get(i, 0.0) + 1.0 / (config.RRF_K + posicion)
            votos.setdefault(i, []).append(cfg.get("nombre") or Path(cfg["ruta"]).parent.name)

    mejores = sorted(puntajes, key=puntajes.get, reverse=True)[:k]
    resultados = []
    for i in mejores:
        fila = metadata.iloc[i]
        resultados.append({
            "doc_id": str(fila["doc_id"]),
            "chunk_id": str(fila["chunk_id"]),
            "fenomeno": int(fila["fenomeno"]),
            "score_rrf": round(puntajes[i], 5),
            "indices_que_lo_encontraron": votos[i],
            "texto": str(fila["texto"]),
            # Campos con los que app/citas.py construye la cita legible.
            **{campo: _valor(fila, campo) for campo in CAMPOS_DE_FUENTE},
        })
    return resultados


# ---------------------------------------------------------------------------
# Filtro tematico para los componentes geograficos del visualizador
# ---------------------------------------------------------------------------
N_TEMA = int(os.getenv("N_TEMA", "600"))

#: Palabras que aparecen en casi toda instruccion de visualizacion y no aportan
#: tema. Sin excluirlas, "muestrame los lugares" se tomaria como consulta de
#: contenido y ensuciaria el filtro.
GENERICAS = {
    "muestrame", "muéstrame", "mapa", "mapas", "lugares", "hablan", "todos", "todas",
    "zonas", "donde", "dónde", "grafica", "gráfica", "concentran", "concentra", "paises",
    "países", "menciones", "mencionan", "afectadas", "afectados", "regiones", "sobre",
    "cuales", "cuáles", "cuantos", "cuántos", "documentos", "tiempo", "evolucion",
    "evolución", "relacion", "relación", "entre", "tendencia", "fuentes", "visualiza",
}


def _palabras_clave(pregunta: str) -> set[str]:
    from ..seguridad import normalizar
    # normalizar() ya quito tildes, asi que basta con el alfabeto latino basico.
    toks = re.findall(r"[a-z]+", normalizar(pregunta))
    return {t[:6] for t in toks if len(t) >= 5 and t not in GENERICAS}


def buscador_tema(pregunta: str, fenomeno: Optional[int] = None) -> set[int]:
    """Conjunto de fragmentos relacionados con la pregunta, para acotar un mapa.

    Sin este filtro, "¿donde operan los grupos armados?" pintaria TODOS los lugares
    del fenomeno 3 en vez de los asociados al tema preguntado. Devuelve un conjunto
    vacio cuando el tema no aparece en el corpus, para que el mapa avise en lugar
    de mostrar un resultado enganoso.
    """
    from ..seguridad import normalizar
    puntajes: dict[int, float] = {}
    for cfg in INDICES:
        vec = cfg["embedder"].encode([cfg["prefijo"] + pregunta],
                                     normalize_embeddings=cfg["normalizar"]).astype("float32")
        _, ids = cfg["indice"].search(vec, N_TEMA * 3)
        posicion = 0
        for i in ids[0]:
            if i < 0 or (fenomeno and FENOMENO_POR_FILA[i] != fenomeno):
                continue
            posicion += 1
            puntajes[int(i)] = puntajes.get(int(i), 0.0) + 1.0 / (config.RRF_K + posicion)
    top = sorted(puntajes, key=puntajes.get, reverse=True)[:N_TEMA]

    claves = _palabras_clave(pregunta) if not fenomeno else set()
    if claves:   # si ninguna palabra clave aparece en los 50 mejores, el tema no esta en el corpus
        textos = [normalizar(t) for t in metadata["texto"].iloc[top[:50]].astype(str)]
        if not any(k in t for k in claves for t in textos):
            return set()
    return set(top)


def estado() -> dict:
    """Resumen para el healthcheck y el endpoint /salud."""
    return {
        "fragmentos": int(len(metadata)),
        "indices": [c.get("nombre") or Path(c["ruta"]).parent.name for c in INDICES],
        "fenomenos": sorted(int(f) for f in set(FENOMENO_POR_FILA.tolist())),
    }
