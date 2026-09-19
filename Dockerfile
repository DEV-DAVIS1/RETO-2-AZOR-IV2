# syntax=docker/dockerfile:1
# ============================================================================
# CODEFEST AD ASTRA 2026 - Equipo Azor IV - imagen del sistema multi-agente
# Build pack en Coolify: Dockerfile. Puerto interno: 8000. Healthcheck: /salud
# ============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- dependencias (capa cacheable) -----------------------------------------
# torch primero desde el indice CPU: la rueda CUDA de PyPI pesa >2 GB y arrastra
# librerias NVIDIA que el contenedor no usa.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- encoders en cache dentro de la imagen ---------------------------------
# OJO: sin heredoc. Coolify inyecta sus `ARG` despues de cualquier linea que
# empiece por "from" (sin distinguir mayusculas), y un `from ... import` dentro
# de un heredoc de Python rompe el script con SyntaxError.
RUN python -c "import sentence_transformers as st; [st.SentenceTransformer(m, device='cpu') for m in ('BAAI/bge-m3', 'intfloat/multilingual-e5-large', 'intfloat/multilingual-e5-large-instruct')]; print('encoders en cache')"

# ---- base vectorial --------------------------------------------------------
# Opcion A (por defecto): llega con el clon (Git LFS). Se verifica que NO sean
# punteros: un puntero pesa ~130 bytes, el indice real ~618 MB.
#
# Opcion B (si Coolify no resuelve LFS): publicar los 4 archivos como assets de
# un GitHub Release y pasar BASE_URL como build arg. Con BASE_URL definido, se
# descargan en el build y se ignora la copia del repo.
ARG BASE_URL=""
COPY base_vectorial/ /datos/base_vectorial/
RUN set -eu; \
    if [ -n "$BASE_URL" ]; then \
        echo "Descargando base vectorial desde $BASE_URL"; \
        for enc in bge e5 e5_instruct; do \
            mkdir -p /datos/base_vectorial/encoder_$enc; \
            curl -fL --retry 3 -o /datos/base_vectorial/encoder_$enc/index.faiss \
                 "$BASE_URL/${enc}_index.faiss"; \
        done; \
        curl -fL --retry 3 -o /datos/base_vectorial/Metadata_AZOR_IV.jsonl \
             "$BASE_URL/Metadata_AZOR_IV.jsonl"; \
    else \
        cp /datos/base_vectorial/encoder_bge/Metadata_AZOR_IV.jsonl /datos/base_vectorial/Metadata_AZOR_IV.jsonl; \
    fi; \
    fallo=0; \
    for f in /datos/base_vectorial/Metadata_AZOR_IV.jsonl \
             /datos/base_vectorial/encoder_bge/index.faiss \
             /datos/base_vectorial/encoder_e5/index.faiss \
             /datos/base_vectorial/encoder_e5_instruct/index.faiss; do \
        if [ ! -f "$f" ]; then echo "FALTA: $f"; fallo=1; continue; fi; \
        tam=$(stat -c%s "$f"); \
        if [ "$tam" -lt 1000000 ]; then \
            echo "PUNTERO LFS SIN RESOLVER: $f ($tam bytes)"; fallo=1; \
        else \
            echo "OK $f ($((tam/1000000)) MB)"; \
        fi; \
    done; \
    if [ "$fallo" -ne 0 ]; then \
        echo "============================================================"; \
        echo "La base vectorial no llego completa. Opciones:"; \
        echo "  1. Subir index.faiss x3 + Metadata_AZOR_IV.jsonl como assets de un"; \
        echo "     GitHub Release y definir el build arg BASE_URL en Coolify."; \
        echo "  2. Montar la base como volumen en /datos/base_vectorial y quitar"; \
        echo "     el COPY de arriba."; \
        echo "============================================================"; \
        exit 1; \
    fi

# ---- codigo ----------------------------------------------------------------
COPY app/ /app/app/
COPY frontend/ /app/frontend/
COPY agent_card.json /app/agent_card.json

RUN useradd --create-home --uid 10001 azor \
    && mkdir -p /app/estado \
    && chown -R azor:azor /app/estado /opt/hf
USER azor

# Rutas que lee app/config.py
ENV BASE_VECTORIAL=/datos/base_vectorial \
    METADATA_PATH=/datos/base_vectorial/Metadata_AZOR_IV.jsonl \
    DIR_ESTADO=/app/estado \
    PORT=8000

EXPOSE 8000

# start-period largo: cargar 3 indices FAISS + 3 encoders tarda minutos en frio.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -fsS http://localhost:8000/salud || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
