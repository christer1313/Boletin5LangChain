Ejercicio 3: TDD con LangGraph

Descripción
 - Este ejemplo implementa un bucle TDD que genera tests (por un LLM), genera código (por otro LLM), ejecuta los tests y repite hasta que pasen todos o se alcance el máximo de intentos.

Estado y grafo
 - El estado se modela como un diccionario por intento con las claves: `tests` (lista), `code` (texto), `passed` (número de tests pasados), `errors` (lista de errores o cadena de compilación).
 - El grafo se guarda como JSON en `graph.json` con nodos por intento y aristas secuenciales. Puedes visualizarlo creando un grafo DOT a partir de ese JSON.

Uso
 - Instala dependencias:

```bash
pip install -r ejercicio3-langgraph/requirements.txt
```

- Ejecutar ejemplo (usa `HUGGINGFACE_API_KEY` y variables para modelos):

```bash
HUGGINGFACE_API_KEY=... \
HUGGINGFACE_MODEL_TESTS="mistralai/Mistral-7B-Instruct" \
HUGGINGFACE_MODEL_CODE="bigcode/starcoder" \
python ejercicio3-langgraph/tdd_langgraph.py --sample-size 1
```

Notas
 - Modelos por defecto seleccionados y razones:
	 - **Tests / razonamiento:** `mistralai/Mistral-7B-Instruct` — buen equilibrio costo/beneficio y capacidad de razonamiento para generar casos límite y contraejemplos.
	 - **Código:** `bigcode/starcoder` — optimizado para generación de código, produce implementaciones sintácticamente correctas y limpias.
 - Temperaturas recomendadas:
	 - Tests: `TEMPERATURE_TESTS` ≈ 0.4–0.7 (más diversidad y razonamiento)
	 - Código: `TEMPERATURE_CODE` ≈ 0.0–0.2 (determinismo)
 - Variables de entorno válidas:
	 - `HUGGINGFACE_API_KEY`: token de HF (si necesario)
	 - `HUGGINGFACE_MODEL_TESTS`: modelo para generación de tests
	 - `HUGGINGFACE_MODEL_CODE`: modelo para generación de código
 - El script genera archivos `graph_{task}.json` y `history_{task}.json` en el directorio del script, y `summary.json` con un resumen por tarea.
