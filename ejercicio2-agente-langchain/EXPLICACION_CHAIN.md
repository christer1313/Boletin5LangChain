# Explicación: Chain de query_with_schema

## Contexto
En este ejercicio hay una sola tool: `query_with_schema`.
Esa tool encapsula una secuencia lógica de acciones y, dentro de algunos pasos,
usa subcadenas LCEL para invocar el modelo.

La idea importante es:
- La tool define la orquestación general.
- LCEL se usa en subpasos donde participa el LLM.

## Flujo real (5 pasos)

### 1. Inferir tabla relevante
Se listan tablas disponibles y se usa una subcadena LCEL para seleccionar la más adecuada según la consulta.

Patrón usado:
```python
prompt | model | StrOutputParser()
```

### 2. Obtener esquema
Se consulta `PRAGMA table_info(...)` sobre la tabla inferida.
Este paso es determinístico (sin LLM).

### 3. Generar SQL
Se usa otra subcadena LCEL para construir una SELECT basada en esquema + tarea.

Patrón usado:
```python
prompt | model | StrOutputParser() | extract_code
```

`extract_code` extrae el SQL cuando el modelo responde dentro de un bloque markdown.

### 4. Validar SQL
Se comprueba que la consulta sea `SELECT`.
Paso determinístico para reducir riesgo de queries destructivas.

### 5. Ejecutar y devolver resultado
Se ejecuta el SQL validado y se devuelve salida formateada.
Paso determinístico.

## Qué es un Runnable
Un `Runnable` es una unidad ejecutable de LangChain con una interfaz de entrada/salida.
Ejemplos típicos:
- Prompt templates
- Modelos
- Parsers
- Composiciones de varios bloques

LCEL encadena runnables con `|`.

## Diferencia entre Tool y pipeline LCEL
- **Tool:** interfaz que el agente puede invocar. Puede contener lógica completa, no solo LLM.
- **Pipeline LCEL:** composición de runnables, muy útil para pasos de prompting/inferencia.

En este proyecto:
- `query_with_schema` = tool de alto nivel.
- `_infer_table` y `_generate_sql` = subpasos con LCEL.
- `_get_schema`, validación y ejecución = pasos determinísticos.

## Dónde entra LangGraph
Si el flujo crece y necesitas:
- ramas condicionales,
- estado persistente,
- reintentos,
- bucles,

LangGraph es una mejor opción para orquestar.
Aun así, cada nodo puede seguir usando LCEL internamente.
