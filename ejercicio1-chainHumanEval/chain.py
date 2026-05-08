"""
Definición de la chain LangChain para generar código desde especificaciones HumanEval.

Este módulo contiene la lógica de la chain LCEL que:
1. Recibe una especificación de tarea en lenguaje natural
2. Invoca un modelo LLM para generar código
3. Extrae el código desde bloques Markdown
4. Devuelve código Python listo para ejecutar
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


def extract_code(text: str) -> str:
    """
    Extrae código Python desde el primer bloque de código Markdown.
    
    Si el modelo devuelve: ```python\ndef foo():\n    pass\n```
    Esta función extrae solo: def foo():\n    pass
    
    Args:
        text: El texto devuelto por el modelo LLM.
        
    Returns:
        El código Python limpio, sin envoltura Markdown.
        Si no hay bloque Markdown, devuelve el texto limpio como está.
    """
    match = re.search(r"```(?:[^\n`]*)\n(?P<code>.*?)```", text, re.DOTALL)
    if not match:
        return text.strip()
    code = match.group("code").strip()
    return code


def build_chain(model: ChatOllama):
    """
    Construye la chain LCEL (LangChain Expression Language) completa.
    
    Flujo de la chain:
    1. prompt: Formatea el input del usuario con instrucciones al modelo
    2. model: Invoca el LLM (ChatOllama) para generar código
    3. StrOutputParser: Convierte la salida a texto limpio
    4. extract_code: Extrae el código desde el bloque Markdown
    
    Cómo se recibe la query:
    - La chain se invoca con: chain.invoke({"query": task["prompt"]})
    - task["prompt"] es un string con la especificación de HumanEval
    - El diccionario {"query": "..."} alimenta el template donde aparece {query}
    - El template sustituye {query} con el contenido de task["prompt"]
    - El resultado formateado se envía al LLM
    
    Ejemplo:
    Input:  chain.invoke({"query": "def has_close_elements(numbers: List..."})
    Template: "You are an expert...\\n\\ndef has_close_elements(...)"
    LLM genera: "```python\\n    for i in...\\n```"
    Output: "    for i in..." (código limpio)
    
    Args:
        model: Instancia de ChatOllama ya configurada.
        
    Returns:
        Un Runnable (chain) que toma {"query": "..."} y devuelve código Python limpio.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Python programmer. "
                "Complete the HumanEval task by writing only the missing Python continuation. "
                "Do not repeat the prompt. Wrap the answer in a markdown code block.",
            ),
            ("human", "{query}"),
        ]
    )
    return prompt | model | StrOutputParser() | RunnableLambda(extract_code)


def build_model(temperature: float) -> ChatOllama:
    """
    Crea e inicializa la instancia del modelo LLM (ChatOllama).
    
    Configura el modelo desde variables de entorno:
    - OLLAMA_MODEL: Nombre del modelo (default: gpt-oss:20b-cloud)
    - OLLAMA_BASE_URL: URL de la API (opcional, usa configuración local si no se especifica)
    - OLLAMA_API_KEY: Token de autenticación (opcional)
    
    Args:
        temperature: Temperatura del modelo (cercana a 0 = más determinista).
        
    Returns:
        Instancia configurada de ChatOllama lista para usarse en la chain.
    """
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
    base_url = os.getenv("OLLAMA_BASE_URL")
    api_key = os.getenv("OLLAMA_API_KEY")

    model_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
    }
    if base_url:
        model_kwargs["base_url"] = base_url
    if api_key:
        model_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return ChatOllama(**model_kwargs)
