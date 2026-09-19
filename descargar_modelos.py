from sentence_transformers import SentenceTransformer

modelos = (
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
)

for m in modelos:
    print(f"Descargando {m}...")
    SentenceTransformer(m)
