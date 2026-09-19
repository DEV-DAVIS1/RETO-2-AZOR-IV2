 
# CODEFEST AD ASTRA 2026 - Etapa 2 - Reto 1 - Equipo Azor IV
# Imagen autosuficiente del agente (Anexo A.4). Build pack en Coolify: Dockerfile.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

# curl: healthcheck. git y git-lfs: la base vectorial viaja en Git LFS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa de dependencias primero: cambiar el codigo no obliga a reinstalar todo.
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.14.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pesos de los encoders dentro de la imagen.
RUN python -c "from sentence_transformers import SentenceTransformer; [SentenceTransformer(m) for m in ('BAAI/bge-m3', 'intfloat/multilingual-e5-large', 'intfloat/multilingual-e5-large-instruct')]"

COPY . .

# El servicio no necesita privilegios: escribe unicamente en /app/estado.
RUN useradd --create-home --uid 10001 azor \
    && mkdir -p /app/estado \
    && chown -R azor:azor /app/estado /opt/hf

USER azor

EXPOSE 8000

# start-period amplio: cargar tres indices FAISS y sus encoders tarda varios
# minutos en frio, y hasta entonces el contenedor no debe declararse enfermo.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8000/salud || exit 1

# Un solo worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
