# CODEFEST AD ASTRA 2026 · Reto 1 — Asistente conversacional

**Equipo Azor IV** · Etapa 2 · Universidad de los Andes / Fuerza Aeroespacial Colombiana

Sistema multi-agente que responde preguntas en lenguaje natural sobre un corpus de
tres fenómenos, comparándolos entre sí y generando visualizaciones, siempre con la
evidencia (`doc_id`, `chunk_id`) que sustenta cada afirmación.

| Fenómeno | Tema |
|---|---|
| 1 | Inteligencia artificial en entornos militares |
| 2 | Seguridad espacial y órbita baja terrestre |
| 3 | Dinámicas territoriales en América Latina |

Base de conocimiento de la Etapa 1: **150 941 fragmentos**, tres índices FAISS
fusionados con Reciprocal Rank Fusion.

---

## 1. Arquitectura en una imagen

```
                        pregunta del usuario
                                 │
                 ┌───────────────▼────────────────┐
                 │  memoria (lectura) + filtro    │   código, sin LLM
                 │  session_id opcional           │   un ataque cuesta 0 tokens
                 └───────────────┬────────────────┘
                                 │
                 ┌───────────────▼────────────────┐        ┌──────────────────┐
                 │  1 · ORQUESTADOR               │────────│  buscar_corpus   │
                 │  gpt-oss-20b · elige 1 rama    │        │  FAISS + metadata│
                 └───┬─────────┬─────────┬────────┘        └──────────────────┘
                     │         │         │        │
        ┌────────────▼──┐ ┌────▼─────┐ ┌─▼──────────┐ ┌──▼─────────────┐
        │ respuesta fija│ │2·REDACTOR│ │3·COMPARADOR│ │4·VISUALIZADOR  │
        │ fuera de      │ │incl.     │ │2–3         │ │gráficos y mapas│
        │ dominio       │ │cronología│ │fenómenos   │ │                │
        └────────────┬──┘ └────┬─────┘ └─┬──────────┘ └──┬─────────────┘
                     │         │         │               │
                     │         │         │      ┌────────▼─────────┐
                     │         │         │      │ consultar_datos  │
                     │         │         │      │ metadata real    │
                     │         │         │      └────────┬─────────┘
                 ┌───▼─────────▼─────────▼───────────────▼────────┐
                 │  memoria (escritura) + log JSONL               │  código, sin LLM
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            respuesta JSON (sección 2.4)
```

El rationale de cada decisión está en **[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md)**.

---

## 2. Entregables y dónde están

| Entregable (sección del PDF) | Ubicación |
|---|---|
| Endpoint del agente (§2.1, §2.4) | `POST /chat` — `app/main.py` |
| Frontend de chat (§1.4, §2.2) | `frontend/index.html` |
| Ficha del agente / agent card (§2.3) | `agent_card.json`, servida en `GET /agent-card` |
| Documento de arquitectura (§1.4) | `docs/ARQUITECTURA.md` |
| Instrucciones de despliegue (§1.4) | este README, sección 5 |
| Dashboard del Reto 2 (§3) | `frontend/dashboard.html` |

---

## 3. Estructura del repositorio

```
app/                     código del sistema
├── main.py              API HTTP: /chat, /agent-card, /salud
├── config.py            configuración por variables de entorno
├── contrato.py          construcción del JSON de la §2.4 y suma de tokens
├── memoria.py           memoria conversacional + log de trazabilidad
├── seguridad.py         filtro determinista anti prompt-injection
├── llm.py               cliente único hacia el gateway de ADL
└── agentes/
    ├── orquestador.py   agente 1 · enrutamiento
    ├── redactor.py      agente 2 · RAG citado
    ├── comparador.py    agente 3 · comparación con evidencia pareada
    ├── visualizador.py  agente 4 · componentes visuales
    └── herramientas.py  tools buscar_corpus y consultar_datos

frontend/
├── index.html           chat del subdominio frontagent
└── dashboard.html       tablero del Reto 2

pruebas/                 bancos de prueba y resultados medidos
utilidades/              herramientas de desarrollo (inspección, mapas, verificación)
base_vectorial/          base de conocimiento de la Etapa 1 (Git LFS)
estado/                  logs y memoria en ejecución (no versionado)
```

---

## 4. Ejecución local

Requiere Python 3.11 y la base vectorial en `base_vectorial/`.

```bash
git lfs install && git lfs pull
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configura el gateway y levanta el servicio:

```bash
export LLM_BASE_URL="https://litellm.admin-adl.codefest2026.augusta.avaldigitallabs.com"
export LLM_API_KEY="TU_API_KEY"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

En PowerShell:

```powershell
$env:LLM_BASE_URL = "https://litellm.admin-adl.codefest2026.augusta.avaldigitallabs.com"
$env:LLM_API_KEY  = "TU_API_KEY"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

El primer arranque tarda varios minutos: carga 150 941 fragmentos, tres índices FAISS
y tres modelos de embeddings. `GET /salud` responde cuando el sistema está listo.

Para desarrollo sin gateway, apunta `LLM_BASE_URL` a Ollama
(`http://localhost:11434/v1`) y ajusta los modelos a los que tengas descargados.

---

## 5. Despliegue en Coolify

Procedimiento del Anexo A de la especificación.

1. **Llave SSH** — Coolify → *Root Team → Keys & Tokens → Private Keys → New
   Private Key → Generate ED25519*. Registra la llave pública en GitHub, en
   *Settings → SSH and GPG keys*.
2. **Aplicación** — *Applications → Private Git Repository (with Deploy Key)*.
3. **Repositorio** — URL del repositorio privado, rama `main`, **Build pack:
   `Dockerfile`**.
4. **Dominio** — *Domains → + Add Domain*:
   `https://agent.azoriv.codefest2026.augusta.avaldigitallabs.com`, **Port
   interno 8000**, *www redirect* en **No redirect**.
   El frontend de chat va en `frontagent.azoriv…` y el dashboard en `dashboard.azoriv…`.
5. **Variables de entorno** — *Environment Variables → + Add*, marcadas para
   *Runtime*:

| Variable | Valor | Obligatoria |
|---|---|---|
| `LLM_API_KEY` | API key de ADL | **sí** |
| `LLM_BASE_URL` | URL del gateway LiteLLM | **sí** |
| `EQUIPO` | `azoriv` | no |
| `MODELO_ORQUESTADOR` | `openai.gpt-oss-20b-1:0` | no |
| `MODELO_REDACTOR` | `meta.llama3-3-70b-instruct` | no |
| `MODELO_COMPARADOR` | `meta.llama3-3-70b-instruct` | no |
| `MODELO_VISUALIZADOR` | `openai.gpt-oss-20b-1:0` | no |
| `CORS_ORIGINS` | dominios del frontend | no |
| `RAG_INDICES` | `bge,e5,e5_instruct` | no |

Las que no son obligatorias ya traen el valor correcto por defecto; se exponen
para poder ajustar el sistema sin reconstruir la imagen.

> **Git LFS.** La base vectorial pesa ~2 GB y viaja en Git LFS. El entorno que
> clona el repositorio debe tener `git-lfs` instalado, o llegarán punteros de
> texto en lugar de los índices. El sistema detecta ese caso al arrancar y lo
> reporta explícitamente. Ten en cuenta que la cuota gratuita de LFS en GitHub es
> de 1 GB: para 2 GB hace falta contratar un *data pack*. Alternativa sin LFS:
> excluir `base_vectorial/` del `.dockerignore`, montarla como volumen en Coolify
> y apuntar `BASE_VECTORIAL` al punto de montaje.

**Healthcheck.** El `Dockerfile` declara `GET /salud` con `start-period` de
10 minutos, porque hasta que los índices no están cargados el contenedor no debe
declararse sano.

---

## 6. Uso del sistema

### `POST /chat`

```bash
curl -X POST https://dashboard.azor-iv.codefest2026.augusta.avaldigitallabs.com/ \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué riesgos genera la basura espacial en la órbita baja?"}'
```

| Campo | Tipo | Descripción |
|---|---|---|
| `pregunta` | string | **obligatorio**. Consulta en lenguaje natural. |
| `session_id` | string | opcional. Hilo conversacional; habilita preguntas de seguimiento. |
| `fenomeno` | 1–3 | opcional. Fuerza el fenómeno en vez de inferirlo. |
| `doc_ids` | lista | opcional. Documentos de contexto del turno anterior. |

Respuesta (formato de la §2.4, abreviada):

```json
{
  "respuesta": "Los desechos espaciales en la órbita baja generan… [13b67ce6657d4fbb_178_1]",
  "evaluacion": {
    "input": "¿Qué riesgos genera la basura espacial en la órbita baja?",
    "actual_output": "Los desechos espaciales…",
    "retrieval_context": ["…", "…"],
    "tools_called": [{"name": "buscar_corpus",
                      "input_parameters": {"query": "…", "fenomeno": 2},
                      "output": ["doc:chunk", "…"]}]
  },
  "metadata": {
    "num_interacciones": 2,
    "agentes_invocados": ["orquestador", "redactor"],
    "tokens": {"input": 2058, "output": 277, "total": 2335},
    "tokens_por_agente": [
      {"agente": "orquestador", "modelo": "gpt-oss-20b", "input": 384, "output": 83, "total": 467},
      {"agente": "redactor", "modelo": "llama-3.3-70b-instruct", "input": 1674, "output": 194, "total": 1868}
    ],
    "latencia_ms": 13790,
    "estado": "ok"
  },
  "traza": {"rama": "redactor", "fenomenos": [2], "session_id": "…"}
}
```

`traza`, `visualizacion`, `puntos`, `grafico` y `evidencia` son extensiones propias
para el frontend y el dashboard; no alteran los tres bloques que evalúa ADL.

### Otros endpoints

| Endpoint | Uso |
|---|---|
| `GET /salud` | Healthcheck: estado del corpus y modelos configurados. |
| `GET /agent-card` | Ficha del sistema multi-agente. |
| `GET /docs` | Documentación interactiva de la API (OpenAPI). |
| `POST /consulta` | Alias de `/chat` que consume el dashboard del Reto 2. |

### Preguntas de seguimiento

Enviando el mismo `session_id`, la memoria resuelve las referencias al turno
anterior **sin gastar tokens adicionales**:

```
→ "¿Qué riesgos genera la basura espacial en la órbita baja?"   rama redactor, fenómeno 2
→ "muéstrame eso en un mapa"                                     rama visualizador, hereda el fenómeno 2
```

---

## 7. Pruebas

```bash
python -m pruebas.probar_contrato       # formato §2.4 y coherencia de la ficha (sin gateway)
python -m pruebas.probar_seguridad      # 15 ataques y 12 preguntas legítimas (sin gateway)
python -m pruebas.probar_orquestador    # precisión de enrutamiento (requiere gateway)
python -m pruebas.probar_comparador     # evidencia pareada válida (requiere gateway)
python -m pruebas.probar_visualizador   # componentes; --sin-llm no consume tokens
```

Resultados medidos y comentados en **[docs/RESULTADOS.md](docs/RESULTADOS.md)**;
las salidas crudas quedan en `pruebas/resultados/`.

---

## 8. Control de presupuesto

La bolsa del equipo es de 100 USD (§1.3). El sistema lee el costo real que reporta
el gateway en la cabecera `x-litellm-response-cost` y lo acumula en
`estado/gasto_acumulado.json`:

```json
{"usd": 0.0218, "llamadas": 12, "tokens": 45210, "llamadas_sin_costo": 0}
```
