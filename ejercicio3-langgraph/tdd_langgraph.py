#!/usr/bin/env python3
"""
TDD con LangGraph conectado a HumanEval.

Este script usa Ollama (local o cloud) para generar tests y código.

Requisitos:
 - Ollama ejecutándose (localmente en localhost:11434 o en cloud)
 - Si es local: modelos descargados (ollama pull llama2 && ollama pull codellama)
 - Si es cloud: URL base y API key (si es requerida)

Configuración por ENV (opcional):
 - OLLAMA_BASE_URL: URL base de Ollama (default http://localhost:11434)
 - OLLAMA_API_KEY: Token de autenticación para Ollama Cloud (opcional)
 - TESTS_MODEL: modelo para generar tests (default llama2)
 - CODE_MODEL: modelo para generar código (default codellama)
 - TEMPERATURE_TESTS: temperatura para tests (default 0.6)
 - TEMPERATURE_CODE: temperatura para código (default 0.0)
 - HUMANEVAL_SAMPLE_SIZE: cantidad de tareas (default 1)
 - MAX_ATTEMPTS: máximo de reintentos (default 5)
"""

import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

import multiprocessing

# Evitar que se use el método 'spawn' (que requiere que los módulos sean
# importables por nombre). Forzar 'fork' en Linux preserva el estado del
# proceso padre y evita errores como "ModuleNotFoundError: No module named 'eh_eval'"
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    # Ya establecido en este proceso, ignorar.
    pass

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))


# Estado TDD: mantiene el contexto del bucle TDD
class TDDState(TypedDict):
    task_id: str          # Identificador de tarea HumanEval
    spec: str             # Especificación (prompt de HumanEval)
    entry_point: str      # Nombre de la función
    tests_source: str     # Tests generados (aserciones)
    code: str             # Código generado
    last_error: str       # Último error (compilación o test), "" si OK
    attempts: int         # Número de intentos
    history: List[Dict]   # Historial de intentos
    error_counts: Dict[str, int]  # Conteo de errores por tipo
    error_history: List[Dict[str, str]]  # Historial detallado de errores
    best_code: str        # Mejor código encontrado
    best_passed: int      # Intentos exitosos del mejor código


def _classify_error(error_msg: str) -> str:
    """Clasifica errores para retroalimentar mejor al generador de código."""
    if not error_msg:
        return "None"
    if error_msg == "Timeout":
        return "Timeout"
    if error_msg == "No result returned":
        return "NoResultReturned"

    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(Error|Exception))", error_msg)
    if match:
        return match.group(1)
    return "RuntimeError"


def _build_error_feedback(state: TDDState) -> str:
    """Construye feedback acumulado con tipos y ejemplos de errores recientes."""
    if not state["error_history"]:
        return ""

    counts = state["error_counts"]
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    counts_block = "\n".join([f"- {err_type}: {count}" for err_type, count in sorted_counts])

    recent = state["error_history"][-3:]
    recent_block = "\n".join(
        [f"- intento {e['attempt']}: [{e['type']}] {e['message']}" for e in recent]
    )

    return (
        "\n\nErrores acumulados detectados en intentos previos:\n"
        f"{counts_block}\n"
        "\nÚltimos errores observados:\n"
        f"{recent_block}\n"
        "\nEvita repetir estos fallos. Si aparece TypeError por tests ambiguos, "
        "prioriza cumplir el comportamiento esperado en los asserts existentes."
    )


# Antes existía una clase `ModelWrapper` para tipado; se elimina porque
# ahora usamos clases separadas para tests y código (`HFTestGenerator`, `HFCodeGenerator`).


# Separar responsabilidades: generador de tests (razonamiento) y generador de código


class OllamaTestGenerator:
    """Genera tests usando la API HTTP de Ollama Cloud (/api/chat)."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        import requests

        self.requests = requests
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        self.api_key = os.getenv("OLLAMA_API_KEY", "")
        self.temperature = float(temperature)

    def _call_http(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = self.requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama call failed with status code {resp.status_code}: {resp.text[:300]}")

        out = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            msg = obj.get("message") or {}
            content = msg.get("content")
            if content:
                out.append(content)
            if obj.get("done"):
                break
        return "".join(out).strip()

    def generate_tests(self, spec: str) -> List[str]:
        prompt = (
            "Eres un modelo que genera casos de prueba.\n\n"
            "Dada la siguiente especificación de una función:\n\n" + spec +
            "\n\nGenera exactamente 5 aserciones de Python (`assert ...`) que cubran casos normales y bordes. "
            "No incluyas entradas inválidas de tipos incompatibles salvo que la especificación las pida explícitamente. "
            "Expón solo líneas `assert ...` ejecutables, sin explicaciones ni bloques de código."
        )
        text = self._call_http(prompt)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[:5]

    def generate_code(self, spec: str, tests: List[str]) -> str:
        raise NotImplementedError("OllamaTestGenerator no implementa generate_code; use OllamaCodeGenerator")


class OllamaCodeGenerator:
    """Genera código usando la API HTTP de Ollama Cloud (/api/chat)."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        import requests

        self.requests = requests
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        self.api_key = os.getenv("OLLAMA_API_KEY", "")
        self.temperature = float(temperature)

    def _call_http(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = self.requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama call failed with status code {resp.status_code}: {resp.text[:300]}")

        out = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            msg = obj.get("message") or {}
            content = msg.get("content")
            if content:
                out.append(content)
            if obj.get("done"):
                break
        return "".join(out).strip()

    def generate_tests(self, spec: str) -> List[str]:
        raise NotImplementedError("OllamaCodeGenerator no implementa generate_tests; use OllamaTestGenerator")

    def generate_code(self, spec: str, tests: List[str]) -> str:
        prompt = (
            "Eres un modelo especializado en generación de código Python.\n\n"
            "Dada la siguiente especificación de una función:\n\n" + spec +
            "\n\nY los siguientes tests (aserciones):\n" + "\n".join(tests) +
            "\n\nEscribe únicamente la implementación Python de la función solicitada. Devuelve solo el código válido (sin markdown ni explicaciones). "
            "Prioriza corrección y claridad; evita prints o código de depuración."
        )
        return self._call_http(prompt)


def _load_evaluation_module():
    # Cargar la copia local de evaluation.py del ejercicio 3
    eval_path = os.path.join(os.path.dirname(__file__), "evaluation.py")
    spec = importlib.util.spec_from_file_location("eh_eval", eval_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Nodos del grafo TDD con LangGraph

def generate_tests_node(state: TDDState, llm_tests: Any) -> TDDState:
    """Genera tests una sola vez al inicio (attempts = 0)."""
    if state["attempts"] > 0:
        return state  # Ya existen tests
    print(f"--- Generando tests (tarea {state['task_id']}) ---")
    tests = llm_tests.generate_tests(state["spec"])
    state["tests_source"] = "\n".join(tests)
    print("Tests generados:")
    for line in tests:
        print(f"  {line}")
    return state


def generate_code_node(state: TDDState, llm_code: Any) -> TDDState:
    """Genera código, con feedback si hay error previo."""
    state["attempts"] += 1
    print(f"--- Intento {state['attempts']} ({state['task_id']}) ---")
    
    feedback = ""
    if state["last_error"]:
        feedback = (
            f"\n\nError anterior:\n{state['last_error']}\n"
            + _build_error_feedback(state)
            + "\n\nIntenta de nuevo corrigiendo el error y evitando repetir patrones fallidos."
        )

    spec_with_feedback = state["spec"] + feedback
    code = llm_code.generate_code(spec_with_feedback, state["tests_source"].split("\n"))
    state["code"] = code
    print(f"Código generado (intento {state['attempts']}):\n{code}")
    return state


def check_tests_node(state: TDDState) -> TDDState:
    """Ejecuta tests y reporta resultado."""
    evaluation = _load_evaluation_module()
    tests_list = state["tests_source"].split("\n")
    
    temp_task = {
        "prompt": state["spec"],
        "entry_point": state["entry_point"],
        "test": state["tests_source"],
        "task_id": state["task_id"],
    }
    result = evaluation.evaluate_task(temp_task, state["code"], timeout_seconds=10)
    
    passed = 1 if result.passed else 0
    state["last_error"] = result.error or ""
    error_type = _classify_error(state["last_error"])
    
    print(f"Tests ejecutados: Passed={passed}/{len(tests_list)}")
    if result.error:
        print(f"Error: {result.error}")
    
    # Guardar en historial
    state["history"].append({
        "attempt": state["attempts"],
        "tests": tests_list,
        "code": state["code"],
        "passed": passed,
        "error_type": error_type,
        "errors": [result.error] if result.error else [],
    })

    if state["last_error"]:
        state["error_counts"][error_type] = state["error_counts"].get(error_type, 0) + 1
        state["error_history"].append(
            {
                "attempt": str(state["attempts"]),
                "type": error_type,
                "message": state["last_error"],
            }
        )
    
    # Actualizar mejor solución
    if passed > state["best_passed"]:
        state["best_passed"] = passed
        state["best_code"] = state["code"]
    
    return state


def route_after_check(state: TDDState) -> Literal["generate_code", END]:
    """Decide si continuar generando código o terminar."""
    if state["last_error"] == "":
        print("✓ Todos los tests pasaron. Solución encontrada.")
        return END
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"✗ Máximo de intentos ({MAX_ATTEMPTS}) alcanzado.")
        return END
    print("Reintentando con feedback...\n")
    return "generate_code"


def build_tdd_graph(llm_tests: Any, llm_code: Any):
    """Construye el grafo LangGraph para TDD."""
    builder = StateGraph(TDDState)
    
    # Añadir nodos
    builder.add_node("generate_tests", lambda state: generate_tests_node(state, llm_tests))
    builder.add_node("generate_code", lambda state: generate_code_node(state, llm_code))
    builder.add_node("check_tests", lambda state: check_tests_node(state))
    
    # Añadir aristas
    builder.add_edge(START, "generate_tests")
    builder.add_edge("generate_tests", "generate_code")
    builder.add_edge("generate_code", "check_tests")
    builder.add_conditional_edges("check_tests", route_after_check)
    
    return builder.compile()


def main():
    # Valores por defecto (modificar en README si quieres cambiarlos)
    DEFAULT_SAMPLE_SIZE = int(os.getenv("HUMANEVAL_SAMPLE_SIZE", "1"))
    HUMANEVAL_SEED = os.getenv("HUMANEVAL_SEED")
    # Recomendado para tests: gpt-oss:120b-cloud por capacidad de razonamiento
    DEFAULT_TESTS_MODEL = os.getenv("TESTS_MODEL", "gpt-oss:120b-cloud")
    DEFAULT_CODE_MODEL = os.getenv("CODE_MODEL", "gpt-oss:120b-cloud")
    DEFAULT_TESTS_TEMP = float(os.getenv("TEMPERATURE_TESTS", "0.6"))
    DEFAULT_CODE_TEMP = float(os.getenv("TEMPERATURE_CODE", "0.0"))

    sample_size = DEFAULT_SAMPLE_SIZE
    tests_temp = DEFAULT_TESTS_TEMP
    code_temp = DEFAULT_CODE_TEMP

    print(f"Conectando a Ollama en {os.getenv('OLLAMA_BASE_URL', 'https://ollama.com')}")
    print(f"Modelo para tests: {DEFAULT_TESTS_MODEL}")
    print(f"Modelo para código: {DEFAULT_CODE_MODEL}\n")

    evaluation = _load_evaluation_module()
    sample_seed = int(HUMANEVAL_SEED) if HUMANEVAL_SEED is not None and HUMANEVAL_SEED != "" else None
    sample = evaluation.load_humaneval_sample(sample_size, seed=sample_seed)

    # Crear wrappers Ollama con temperaturas distintas (razonamiento vs código)
    llm_tests = OllamaTestGenerator(DEFAULT_TESTS_MODEL, temperature=tests_temp)
    llm_code = OllamaCodeGenerator(DEFAULT_CODE_MODEL, temperature=code_temp)

    # Construir el grafo LangGraph
    graph = build_tdd_graph(llm_tests, llm_code)

    overall = {"tasks": []}
    out_dir = os.path.dirname(__file__)
    
    for task in sample:
        print(f"\n{'='*60}")
        print(f"Tarea: {task['task_id']}")
        print(f"{'='*60}\n")
        
        # Inicializar estado
        initial_state: TDDState = {
            "task_id": task["task_id"],
            "spec": task["prompt"],
            "entry_point": task["entry_point"],
            "tests_source": "",
            "code": "",
            "last_error": "",
            "attempts": 0,
            "history": [],
            "error_counts": {},
            "error_history": [],
            "best_code": "",
            "best_passed": -1,
        }
        
        # Invocar el grafo
        final_state = graph.invoke(initial_state)
        
        # Guardar resultados
        task_id_safe = task["task_id"].replace("/", "_")
        
        # Guardar historial
        with open(os.path.join(out_dir, f"history_{task_id_safe}.json"), "w") as f:
            json.dump(final_state["history"], f, indent=2)
        
        # Guardar resumen por tarea
        task_summary = {
            "task_id": task["task_id"],
            "best_passed": final_state["best_passed"],
            "total_attempts": final_state["attempts"],
            "final_success": final_state["last_error"] == "",
        }
        with open(os.path.join(out_dir, f"summary_{task_id_safe}.json"), "w") as f:
            json.dump(task_summary, f, indent=2)
        
        overall["tasks"].append(task_summary)

    # Guardar resumen general
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n" + "="*60)
    print("Procesado. Resultados guardados en el directorio del script.")
    print("="*60)


if __name__ == "__main__":
    main()
