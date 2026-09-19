# Documento de arquitectura — Reto 1

**Equipo Azor IV** · CODEFEST AD ASTRA 2026 · Etapa 2

Este documento explica el diseño del sistema multi-agente y el *rationale* de cada
decisión, según lo exigido en la sección 1.4 de la especificación. Es el insumo del
Bloque D de la evaluación (§2.5.4).

---

## 1. El problema

Tres fenómenos, un corpus de 150 941 fragmentos y un evaluador que hará preguntas
abiertas. Las preguntas no son homogéneas: "¿qué es la órbita baja?" es una
consulta de recuperación; "¿en qué se parecen la IA militar y la seguridad
espacial?" exige contrastar dos cuerpos documentales sin inventar el puente entre
ellos; "muéstrame un mapa" no pide texto sino un componente visual poblado con
datos reales.

Un único agente RAG atendería las tres, pero mal: el mismo prompt que sirve para
resumir induce a fabricar analogías al comparar, y no produce estructuras de datos
para un tablero. **Cada tipo de pregunta necesita un contrato distinto con el
modelo**, y de ahí sale la arquitectura.

---

## 2. Los agentes

El sistema tiene cuatro agentes: el orquestador exigido como agente principal, los
dos sub-agentes obligatorios (respuesta en lenguaje natural y generación de
visualizaciones) y un cuarto agente propio, el comparador.

### 2.1 Orquestador — `app/agentes/orquestador.py`

**Modelo: gpt-oss-20b.** Recibe la consulta y elige **una sola rama**. No redacta,
no recupera documentos, no dialoga con los sub-agentes: su única salida es una
decisión estructurada.

```json
{"rama": "redactor", "fenomenos": [2], "query": "riesgos basura espacial órbita baja"}
```

*Rationale.* El enrutamiento es una tarea de clasificación con cuatro clases, no de
generación. Medido con `pruebas/probar_orquestador.py`, el modelo más pequeño del
catálogo la resuelve de forma estable, y así el enrutamiento se mantiene en torno
al 15 % de los tokens del turno (467 de 2 335 en el ejemplo del README); el resto
se invierte donde sí cambia la calidad de la respuesta.

*Alternativa descartada.* Un esquema iterativo tipo ReAct, donde el orquestador
consulta, evalúa y vuelve a decidir, mejoraría los casos ambiguos a costa de
multiplicar las interacciones y la latencia — las dos métricas del Bloque B
(§2.5.2), que se normalizan contra el resto de equipos. Se prefirió **una sola
decisión, con una regla de desempate explícita** en el prompt ("ante la duda entre
redactor y comparador, elige redactor") y una validación en código que lleva
cualquier salida malformada a la rama de menor riesgo.

### 2.2 Redactor — `app/agentes/redactor.py`

**Modelo: Llama 3.3 70B.** Responde preguntas sobre un fenómeno, incluida su
evolución cronológica, usando exclusivamente los fragmentos recuperados.

*Rationale del prompt.* Está escrito contra las métricas del Bloque A (§2.5.1):

- **Faithfulness (30 %).** Prohibición explícita de agregar causas, cifras, fechas
  o nombres ausentes del fragmento citado, y una salida fija obligatoria cuando el
  corpus no responde. El sesgo del sistema es **admitir el vacío antes que
  rellenarlo**: una respuesta incompleta pierde algo de Answer Relevancy, una
  inventada pierde Faithfulness y credibilidad ante los expertos.
- **Cita verificable.** Cada afirmación cierra con el `chunk_id` exacto entre
  corchetes. Es lo que permite al evaluador —y al frontend— rastrear cualquier
  frase hasta su origen.
- **Tono (25 %).** Respuesta directa primero, máximo cinco viñetas, 180 palabras,
  español profesional.

*Por qué el modelo grande aquí.* Es el agente que produce el texto que se evalúa
en el 40 % de la nota. La diferencia entre un modelo de 20B y uno de 70B se nota
precisamente en respetar restricciones negativas ("no agregues lo que no está").

### 2.3 Comparador — `app/agentes/comparador.py` *(agente adicional)*

**Modelo: Llama 3.3 70B.** Es el agente que el equipo añadió por encima del mínimo
exigido, y existe por una razón concreta: **comparar es donde un RAG convencional
alucina con más facilidad**. Al pedirle a un modelo que relacione dos fenómenos, su
tendencia es construir el puente aunque los documentos nunca lo hayan trazado.

La defensa es estructural, no un ruego en el prompt:

1. **Recuperación equilibrada.** `buscar_corpus` se ejecuta una vez *por fenómeno*,
   con la consulta enriquecida con el nombre del fenómeno. Sin esto, el fenómeno
   con más documentos domina el ranking global y el otro lado queda sin evidencia.
2. **Evidencia pareada.** El modelo no redacta prosa libre: devuelve JSON con una
   afirmación y una cita **por cada fenómeno y por cada aspecto comparado**.
3. **Validación en código, 0 tokens.** Se descarta todo ítem cuya cita no exista o
   no pertenezca al fenómeno declarado, y toda afirmación evasiva ("no se
   menciona", "puede inferirse") o demasiado corta para ser sustantiva.
4. **Redacción determinista.** El texto final lo arma `armar_texto()` con los ítems
   que sobrevivieron. **El modelo extrae; el código redacta.** Qué fenómeno quedó
   "sin información" también lo decide el código, de modo que la síntesis nunca
   contradice las viñetas que se muestran.

El resultado es que una comparación sin respaldo se reporta como tal —"los
documentos no muestran coincidencias respaldadas en ambos fenómenos"— en lugar de
fabricarse. `pruebas/probar_comparador.py` informa cuántos ítems descartó la
validación: son alucinaciones que el modelo intentó colar y que nunca llegaron al
usuario.

### 2.4 Visualizador — `app/agentes/visualizador.py`

**Modelo: gpt-oss-20b.** Traduce una instrucción en lenguaje natural en un
componente visual poblado con datos reales. Seis componentes disponibles:

| Componente | Pregunta analítica que responde |
|---|---|
| `mapa_puntos` | ¿Dónde se concentran las menciones de este tema? |
| `mapa_coropletico` | ¿Qué países aparecen más en este fenómeno? |
| `linea_tiempo` | ¿Cómo evolucionó este tema en el tiempo? |
| `matriz_calor` | ¿Cómo se cruzan dos dimensiones (país × fenómeno)? |
| `red_coocurrencia` | ¿Qué entidades aparecen juntas en los documentos? |
| `panel_evidencia` | ¿En qué fuentes se basa lo que acabas de decir? |

*Rationale.* El agente hace **dos llamadas al modelo como máximo**, y la del medio
no existe:

1. **Selección** — el modelo elige componente y filtros mediante *tool calling*.
2. **Ejecución** — `generar_visualizacion()`, **determinista, sin LLM**, agrega la
   metadata real.
3. **Descripción** — el modelo describe un resumen compacto del resultado ya
   calculado, no el JSON completo.

Que el paso 2 sea código y no modelo es la decisión central: **el modelo nunca
produce las cifras, solo las describe**, así que no puede inventar un lugar ni un
conteo. Si el modelo no emite un *tool call* válido, una heurística determinista
resuelve la selección con 0 tokens en lugar de fallar.

Dos guardas adicionales sobre los filtros que el modelo propone: los años solo se
aplican si el usuario escribió un año en la pregunta, y un país solo si su nombre
aparece en ella. Sin ellas, el modelo inventa recortes que vacían el componente.

**Filtro temático.** Los componentes geográficos se calculan sobre los fragmentos
relacionados con la pregunta, no sobre el fenómeno completo: sin esto, "¿dónde
operan los grupos armados?" pintaría todos los lugares del fenómeno 3. Cuando el
tema no aparece en el corpus, el filtro devuelve vacío y el componente lo advierte
en vez de mostrar un resultado engañoso.

---

## 3. Las herramientas

### `buscar_corpus` — recuperación

Fusiona los tres índices FAISS de la Etapa 1 con **Reciprocal Rank Fusion**:

```
puntaje(fragmento) = Σ_índices  1 / (60 + posición_en_ese_índice)
```

*Rationale.* Los tres índices se construyeron con modelos de embeddings distintos
(`bge-m3`, `multilingual-e5-large`, `multilingual-e5-large-instruct`), cuyas
escalas de similitud no son comparables entre sí: promediar sus distancias sería
incorrecto. RRF combina **rankings**, no puntajes, y no necesita calibración. Un
fragmento que los tres modelos consideran relevante sube por encima de uno que
solo convence a uno.

Al cargar, el sistema verifica que cada índice tenga tantos vectores como filas de
metadata y que su dimensión coincida con la del modelo declarado: un índice
construido con otro modelo devolvería vecinos sin sentido **en silencio**.

### `consultar_datos` — agregación

Agrega la metadata enriquecida (`geo`, `fechas`, `fenomeno`, entidades) para poblar
los componentes. Es determinista y no invoca ningún modelo. Cada valor conserva los
`doc_id` y `chunk_id` que lo sustentan, según el requisito de trazabilidad (§3.3.3).

No carga el texto de los fragmentos en memoria: guarda los *offsets* de cada línea
del JSONL y lee bajo demanda, solo cuando el componente es `panel_evidencia`.

---

## 4. Memoria y trazabilidad — `app/memoria.py`

Las dos cajas que abren y cierran el flujo son **código sin LLM**, y eso es una
decisión de diseño, no una simplificación.

**Lectura.** Antes del orquestador, la memoria recupera del turno anterior los
`doc_ids` y los fenómenos. Cuando la pregunta es de seguimiento —"muéstrame eso en
un mapa"— el fenómeno se hereda de ahí. Resolver eso con una llamada adicional al
modelo costaría tokens e interacciones, que son exactamente las métricas del
Bloque B; aquí se resuelve con un diccionario.

**Escritura.** Lo que se guarda es estructurado y acotado —rama, fenómenos,
`doc_ids`—, nunca el texto libre del usuario. Así **una inyección de prompt no
puede sobrevivir a través de la memoria hasta el turno siguiente**.

**Log.** Cada consulta deja una línea en `estado/interacciones.jsonl` con rama,
agentes, tokens por agente, latencia y `doc_ids`. Es la traza auditable de la
solución. Un fallo de escritura del log nunca tumba una respuesta ya generada.

---

## 5. Seguridad — `app/seguridad.py`

Tres barreras, ordenadas de más barata a más cara:

1. **Filtro determinista de entrada.** Bloquea cinco familias de ataque —cambio de
   instrucciones, exfiltración de prompt, suplantación de rol, exfiltración de
   credenciales y ejecución de código— antes de gastar un solo token. Normaliza
   mayúsculas, tildes y caracteres invisibles, y detecta la evasión tipográfica
   clásica (`i g n o r a`) compactando el texto cuando encuentra una racha de
   letras sueltas.
2. **El orquestador**, que clasifica como `fuera_dominio` cualquier intento que
   sobreviva al paso 1.
3. **Los prompts de los sub-agentes**, que tratan el texto recuperado como DATOS y
   no como instrucciones, y `sanear_fragmento()`, que desactiva delimitadores de
   rol (`<|im_start|>`, `### Instruction:`) incrustados en el corpus. Esto cierra
   la **inyección indirecta**: un documento del corpus no puede dar órdenes.

*Criterio de calibración.* El filtro solo bloquea patrones de alta confianza. Un
falso positivo cuesta puntos en el Bloque A (40 %), que pesa el doble que el
Bloque C (20 %), y además el orquestador es la segunda barrera. **Ante la duda, se
deja pasar.** El banco `pruebas/probar_seguridad.py` mide las dos caras: 15/15
ataques bloqueados y 12/12 preguntas legítimas admitidas, incluidas varias que
usan deliberadamente el vocabulario de los ataques ("¿qué reglas de enfrentamiento
mencionan los documentos?", "¿qué papel juegan las claves criptográficas?").

Ante un ataque, la respuesta es **única y no repite el texto del atacante** —para
que el ataque no se refleje en la salida— ni dice qué regla se activó, para no dar
pistas con las que afinar el siguiente intento.

---

## 6. El contrato de respuesta — `app/contrato.py`

Todas las ramas salen por el mismo constructor, incluidas las de error, de modo
que **ADL siempre recibe un JSON parseable**, incluso si el gateway de modelos está
caído.

La clase `Contador` existe por el requisito obligatorio de la §2.4:
`metadata.tokens.total` debe reflejar el consumo de todos los modelos. Cada agente
anota su consumo; el total se **deriva** de esas anotaciones y nunca se escribe a
mano, así que no puede desincronizarse. De ahí salen también
`metadata.tokens_por_agente` y `num_interacciones`.

`pruebas/probar_contrato.py` verifica el contrato sin necesidad de gateway, e
incluye una comprobación que evita un error costoso: que el modelo declarado en
`agent_card.json` sea **el mismo** que el desplegado. Si divergen, ADL calcularía
el costo por pregunta con el precio equivocado.

---

## 7. Eficiencia — cómo se gastan los tokens

Medición real del ejemplo del README (rama redactor, gpt-oss-20b local):

| Agente | Entrada | Salida | Total | Peso |
|---|---|---|---|---|
| orquestador | 384 | 83 | 467 | 20 % |
| redactor | 1 674 | 194 | 1 868 | 80 % |
| **total** | **2 058** | **277** | **2 335** | |

Decisiones que sostienen ese perfil:

- **Una sola llamada por agente** en las ramas redactor y comparador; dos como
  máximo en el visualizador.
- **Las ramas más baratas son las que no llaman a ningún modelo**: un ataque
  bloqueado y una pregunta fuera de dominio detectada por el filtro cuestan
  0 tokens, 0 interacciones y ~1 ms.
- **La memoria evita una llamada** en cada pregunta de seguimiento.
- **El visualizador describe un resumen compacto**, no el JSON completo del
  componente, que puede tener cientos de filas.
- `RAG_INDICES` permite degradar a menos índices si hiciera falta recortar
  latencia, sin tocar código.

---

## 8. Despliegue

Imagen Docker autosuficiente (Anexo A.4). Dos decisiones marcan su comportamiento:

- **`torch` desde el índice de solo CPU.** La rueda por defecto arrastra CUDA
  (>2 GB) que el contenedor nunca usa.
- **Los encoders se descargan durante el *build*.** Dejarlos para el arranque haría
  que el primer despliegue bajara pesos desde Hugging Face mientras el healthcheck
  ya está contando.

**Un solo worker de uvicorn**: los índices FAISS y los encoders ocupan varios GB de
RAM y cada worker cargaría su propia copia. La concurrencia la da el bucle
asíncrono.

**`start-period` de 10 minutos** en el healthcheck: hasta que los índices no están
cargados, el contenedor no debe declararse sano.

El servicio corre como usuario sin privilegios y solo escribe en `/app/estado`.
Ninguna credencial vive en el código ni en la imagen: todas se inyectan como
variables de entorno desde Coolify (Anexo A.6), lo que permite rotarlas sin
reconstruir.

---

## 9. Limitaciones conocidas

- **La geocodificación tiene homónimos.** "Florida" o "Cáceres" en un documento
  colombiano se resuelven a Estados Unidos y España. El sistema corrige los casos
  inequívocos con una tabla explícita y descarta el resto mediante *bounding box*
  del país dominante del documento, pero no es exhaustivo.
- **Los conteos son menciones, no medidas del fenómeno.** El prompt del
  visualizador lo dice explícitamente y prohíbe presentarlos como índices de
  riesgo; aun así, es una lectura que un usuario puede hacer por su cuenta.
- **La memoria es de proceso.** Vive en RAM con TTL de una hora y tope de 500
  sesiones. Con varias réplicas del contenedor, las sesiones no se comparten entre
  ellas. Para el alcance de la evaluación —una sesión por evaluador— es suficiente;
  escalar exigiría un almacén externo.
- **La latencia de arranque es alta** (varios minutos) por el tamaño de la base
  vectorial. Es un costo que se paga una vez por despliegue, no por consulta.
