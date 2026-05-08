# Boletin5LangChain

Repositorio para trabajar sobre el boletín 5 de la asignatura de Generación Automática de Código con LangChain/LangGraph.

## Qué hace este proyecto

Este ejercicio automatiza una evaluación tipo HumanEval con una chain de LangChain:

1. Toma la especificación de una tarea del benchmark.
2. Se la envía a un modelo instructivo por medio de la API de Ollama.
3. Extrae el código generado desde un bloque Markdown.
4. Une la completación con el prompt original de HumanEval.
5. Ejecuta los tests públicos de la tarea en un proceso aislado.
6. Calcula el porcentaje de aciertos sobre una muestra aleatoria de, al menos, 30 problemas.

La evaluación no usa los hidden tests oficiales porque no vienen en el dataset descargado; en su lugar, usa los tests públicos disponibles en cada ejemplo, lo que sirve como automatización reproducible para el boletín.

## Estructura

El proyecto está organizado en 3 módulos:

- **chain.py**: Define la chain LLM
  - `extract_code()`: Extrae código desde bloques Markdown
  - `build_chain()`: Construye la cadena LCEL
  - `build_model()`: Configura ChatOllama

- **evaluation.py**: Lógica de evaluación automática
  - `load_humaneval_sample()`: Descarga el dataset
  - `evaluate_task()`: Evalúa una tarea individual
  - `_execute_tests()`: Ejecuta código en proceso aislado

- **humaneval_chain.py**: Script principal de orquestación
  - `main()`: Coordina todo el flujo

- **requirements.txt**: Dependencias mínimas

## Instalación

1. Crea un entorno virtual.
2. Instala las dependencias con pip install -r requirements.txt.
3. Configura las variables de entorno de Ollama si usas la API remota:
	- OLLAMA_MODEL
	- OLLAMA_BASE_URL
	- OLLAMA_API_KEY
export OLLAMA_MODEL="gpt-oss:20b-cloud"
export OLLAMA_BASE_URL="https://ollama.com"
export OLLAMA_API_KEY="tu_api_key_aqui"

Si no defines OLLAMA_BASE_URL, el script usa la URL por defecto de tu instalación local de Ollama.

## Ejecución

Ejemplo:

python humaneval_chain.py --sample-size 30 --seed 42 --temperature 0.0

Parámetros útiles:

- sample-size: número de tareas a evaluar.
- seed: semilla para elegir la muestra.
- temperature: temperatura del modelo, recomendable cerca de 0 para más determinismo.
- timeout: tiempo máximo por tarea en segundos.

## Idea de la chain

### Flujo

`prompt -> ChatOllama -> StrOutputParser -> extract_code`

### Cómo se recibe la query

1. **main()** obtiene una tarea: `task["prompt"] = "def has_close_elements(numbers: List[float], threshold: float) -> bool:..."`
2. **Invoca la chain**: `chain.invoke({"query": task["prompt"]})`
3. **Template sustituye {query}**: El diccionario `{"query": "..."}` alimenta la variable `{query}` del template
4. **Mensaje formateado al LLM**:
   ```
   System: "You are an expert Python programmer..."
   Human: "def has_close_elements(numbers: List[float], threshold: float) -> bool:..."
   ```
5. **LLM genera código**: Devuelve `"```python\n    for i in range(len(numbers)):\n        ...\n```"`
6. **StrOutputParser**: Limpia el texto
7. **extract_code**: Extrae solo `"    for i in range(len(numbers)):\n        ..."`

### Resultado  

El evaluador combina el código extraído con el prompt original para ejecutar los tests:
```python
def has_close_elements(...):  # Del prompt
  for i in range(...):   # Del LLM
    ...

# Luego ejecuta los tests
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
```
