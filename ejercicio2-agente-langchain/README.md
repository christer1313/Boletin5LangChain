# Ejercicio 2 - Agente LangChain

Este proyecto reimplementa el agente del ejercicio 2 usando LangChain con un estilo declarativo basado en `@tool` y `create_agent`.
Al ejecutar `agent_ej2.py`, el programa entra directamente en un REPL interactivo.

## Qué hace

- Expone una sola tool:
  - `query_with_schema`: infiere la tabla relevante a partir de la consulta, obtiene su esquema, genera SQL segura y ejecuta el resultado.
- Usa un modelo alojado en Hugging Face, pensado para tareas de tool calling e instrucciones.
- Arranca siempre en modo interactivo para que puedas conversar con el agente sin pasar argumentos.

## Archivos principales

- `ejercicio2-agente-langchain/agent_ej2.py`: implementación del agente y del REPL.
- `ejercicio2-agente-langchain/agent_ej2_runnable.py`: implementación alternativa donde la tool usa una única pipeline de `Runnable` end-to-end.
- `ejercicio2-agente-langchain/requirements.txt`: dependencias mínimas.

## Requisitos

- Definir la variable de entorno `HF_TOKEN` con tu token de Hugging Face.
- Colocar `Chinook.sqlite` en la misma carpeta que `agent_ej2.py`.

## Instalación

```bash
pip install -r ejercicio2-agente-langchain/requirements.txt
```

## Uso

```bash
export HF_TOKEN="tu_token_aqui"
python ejercicio2-agente-langchain/agent_ej2.py
```

El archivo abre un REPL directamente. Escribe preguntas o instrucciones y usa `exit`, `quit` o `salir` para terminar.


## Decisiones de diseño

- `@tool` permite definir herramientas de forma declarativa y legible.
- `create_agent(...)` simplifica la composición del agente y deja la intención clara: modelo, herramientas y prompt del sistema.
- Se mantiene `HuggingFaceHub` porque en el Boletín 4 ya trabajamos con Hugging Face
- Se usa un modelo instruct/tuned porque mejora la fiabilidad al elegir herramientas y seguir instrucciones.
- El modo REPL deja una experiencia simple: ejecutar y conversar.

## Flujo de la chain: `query_with_schema`

Esta es la herramienta clave que demuestra las ventajas de LangChain frente a smolagents:

**Pasos de la cadena:**
1. **Inferir tabla** — usar LCEL para deducir la tabla más relevante a partir de la query y las tablas disponibles.
2. **Obtener esquema** — recuperar estructura (columnas, tipos) con PRAGMA (`_get_schema`).
3. **Generar SQL** — usar LCEL + `PromptTemplate` para que el LLM genere una SELECT segura a partir del esquema + tarea.
4. **Validar SQL** — comprobar que es una SELECT (no UPDATE, DELETE, etc.).
5. **Ejecutar** — invocar el helper interno de SQL y devolver el resultado.

**Por qué esto aprovecha LangChain:**

- **Separación de pasos:** cada paso es una función independiente. Un error no rompe toda la cadena.
- **LCEL (pipe syntax) + PromptTemplate:** en lugar de concatenar strings o formatear queries manualmente, LangChain compone la cadena con `prompt | model | parser`. Esto es más limpio, legible y reutilizable.
- **Validación intermedia:** el LLM se invoca sólo cuando es necesario (primero para inferir la tabla y luego para generar SQL). El resto de pasos son determinísticos.
- **Escalabilidad:** si luego quieres añadir memoria (recordar tablas consultadas), callbacks (loguear cada paso), o cacheo de esquemas, LangChain ofrece hooks para ello. Con smolagents, tendrías que reescribir más.
- **Reutilización:** los bloques LCEL y sus prompts pueden reutilizarse, adaptarse o extenderse sin tocar la lógica de la herramienta.

## Aclaración conceptual: Runnable, LCEL, tool y LangGraph

- **Runnable:** unidad ejecutable de LangChain que transforma una entrada en una salida.
  Ejemplos: un prompt, un modelo, un parser o una composición de varios.
- **LCEL:** notación de composición (`|`) para encadenar runnables.
  Patrón típico en pasos con LLM: `prompt | model | parser`.
- **Tool:** función expuesta al agente. Puede contener una secuencia lógica completa
  (determinística + pasos con LLM).
- **En este ejercicio:** `query_with_schema` es la tool, y dentro usa subcadenas LCEL
  sólo donde aporta valor (inferencia de tabla y generación de SQL).
- **LangGraph:** opción para flujos más complejos (ramas, estado, reintentos, bucles).
  Puede incluir nodos que internamente usen LCEL.

**Comparación conceptual:**

- **Smolagents:** te da un agente que ejecuta tools y deja que el LLM decida cuál usar según descripción. Es minimalista y funciona bien para casos simples.
- **LangChain:** te permite componer cadenas con pasos explícitos, validaciones, y uso controlado del LLM. Es más verboso pero mucho más flexible.

En este ejercicio, `query_with_schema` es el primer paso hacia arquitecturas más complejas que podrías construir después (p. ej. recuperación de información multi-tabla, rutas condicionales con `LangGraph`).

