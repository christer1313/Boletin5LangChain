# Guía de Agentes Inteligentes con Smolagents y Docker (HF Edition)

Este repositorio contiene la resolución de tres ejercicios prácticos utilizando la librería `smolagents`, orquestados mediante **Docker** y conectados a modelos de lenguaje profesionales a través de la API de **Hugging Face** utilizando **LiteLLM** como puente de conexión.

---

## 🚀 Descubrimiento Técnico: Conexión con Hugging Face

Tras explorar diversas configuraciones, se ha determinado que la forma más estable y eficiente de conectar los agentes con el Inference Hub de Hugging Face en un entorno Docker es mediante la integración de `LiteLLMModel`. 

Esta configuración permite abstraer la complejidad de la API y utilizar modelos de alta capacidad como `Qwen2.5-Coder-32B` sin consumo de recursos locales:

```python
# Configuración final validada
hf_token = os.getenv("HF_TOKEN")

# Guía de Agentes Inteligentes con Smolagents + LiteLLM + Hugging Face (Docker)

Este repositorio contiene la resolución de ejercicios prácticos usando `smolagents`, orquestados mediante **Docker** y conectados a modelos de lenguaje de **Hugging Face** usando **LiteLLM** como puente (vía `LiteLLMModel`).

## 🚀 Descubrimiento técnico: conexión con Hugging Face
Tras probar distintas configuraciones, la opción más estable en Docker fue usar `LiteLLMModel` con `model_id` en formato `huggingface/...` y `HF_TOKEN` como `api_key`.

Ejemplo de configuración:

```python
import os
from smolagents import LiteLLMModel

hf_token = os.getenv("HF_TOKEN")

model = LiteLLMModel(
     model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
     api_key=hf_token,
     temperature=0.2,
)
```

## 📘 Estado de los ejercicios
1. **Agente SQL (Chinook SQLite)** ✅
    - **Estado:** Operativo.
    - **Logro:** inspecciona tablas y ejecuta consultas; se validó detectando que `Chinook.sqlite` contiene registros más recientes (2021–2025) en lugar de históricos de 2009.

2. **Agente de sistema (Bash)** ✅
    - **Estado:** Operativo.
    - **Logro:** ejecuta comandos de shell dentro del contenedor; usa herramientas de bash cuando encuentra restricciones en funciones nativas (p. ej. `open()`).

3. **Agente no relacional (MongoDB)** ⚠️
    - **Estado:** Prototipo funcional / en desarrollo.
    - **Nota:** herramientas con `pymongo` implementadas; la conectividad contenedor↔host en Linux requiere ajuste (red / URI).

## 📅 Roadmap (v2.0)
- **Interfaz interactiva:** REPL para conversar con el agente sin reiniciar el contenedor.
- **Persistencia en MongoDB:** estabilizar el flujo con redes Docker.
- **Manejo de contexto:** mejorar memoria de corto plazo en tareas encadenadas.

## 🛠️ Ejecución

### Requisito: token de Hugging Face

```bash
export HF_TOKEN="tu_token_aqui"
```

### Construir imagen

```bash
docker build -t smolagents-workshop .
```

### Lanzar agente (ejemplo SQL)

```bash
docker run --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  smolagents-workshop \
  python script_sql.py
```

### Recomendado: bind mount para poder editar el código

```bash
docker run --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD":/app \
  -w /app \
  smolagents-workshop \
  python script_sql.py
```

## 🧩 Notas
- Si aparece `ImportError: You must install package ddgs...`, reconstruye la imagen (`docker build --no-cache ...`) o instala dependencias en local con `pip install -r requirements.txt`.