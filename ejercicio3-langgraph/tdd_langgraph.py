#!/usr/bin/env python3
"""
TDD con LangGraph conectado a HumanEval.

Este script permite seleccionar dos modelos distintos (uno para generar tests
y otro para generar código) usando Ollama o la API de Hugging Face Inference.

Configuración por ENV:
 - TESTS_MODEL_TYPE: 'ollama' o 'hf' (default 'ollama')
 - TESTS_MODEL_NAME: nombre del modelo (ollama model o hf model id)
 - CODE_MODEL_TYPE: 'ollama' o 'hf' (default 'ollama')
 - CODE_MODEL_NAME: nombre del modelo
 - TEMPERATURE: temperatura float para ambos modelos (default 0.0)
 - HUGGINGFACE_API_KEY: token (si usa HF)
 - OLLAMA_BASE_URL / OLLAMA_MODEL / OLLAMA_API_KEY para Ollama
"""

import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))


class ModelWrapper:
    def generate_tests(self, spec: str) -> List[str]:
        raise NotImplementedError()

    def generate_code(self, spec: str, tests: List[str]) -> str:
        raise NotImplementedError()


# Separar responsabilidades: generador de tests (razonamiento) y generador de código


class HFBase(ModelWrapper):
    def __init__(self, model_name: str, temperature: float = 0.0):
        import requests

        self.requests = requests
        self.model_name = model_name
        self.api_url = os.getenv("HUGGINGFACE_API_URL") or f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"} if os.getenv("HUGGINGFACE_API_KEY") else {}
        self.temperature = temperature

    def _call(self, prompt: str) -> str:
        payload = {"inputs": prompt, "parameters": {"temperature": float(self.temperature)}}
        resp = self.requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data['error']}")
        if isinstance(data, list):
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            return str(first)
        return str(data)


class HFTestGenerator(HFBase):
    def generate_tests(self, spec: str) -> List[str]:
        prompt = (
            "Eres un modelo que genera casos de prueba.\n\n"
            "Dada la siguiente especificación de una función:\n\n" + spec +
            "\n\nGenera exactamente 5 aserciones de Python (`assert ...`) que cubran casos normales, bordes y entradas inválidas cuando proceda. "
            "Expón solo las líneas de aserciones, sin explicaciones ni bloques de código."
        )
        text = self._call(prompt)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[:5]

    def generate_code(self, spec: str, tests: List[str]) -> str:
        raise NotImplementedError("HFTestGenerator no implementa generate_code; use HFCodeGenerator")


class HFCodeGenerator(HFBase):
    def generate_tests(self, spec: str) -> List[str]:
        raise NotImplementedError("HFCodeGenerator no implementa generate_tests; use HFTestGenerator")

    def generate_code(self, spec: str, tests: List[str]) -> str:
        prompt = (
            "Eres un modelo especializado en generación de código Python.\n\n"
            "Dada la siguiente especificación de una función:\n\n" + spec +
            "\n\nY los siguientes tests (aserciones):\n" + "\n".join(tests) +
            "\n\nEscribe únicamente la implementación Python de la función solicitada. Devuelve solo el código válido (sin markdown ni explicaciones). "
            "Prioriza corrección y claridad; evita prints o código de depuración."
        )
        return self._call(prompt).strip()


def _load_evaluation_module():
    # Cargar el módulo evaluation.py del ejercicio1 usando su ruta
    base = os.path.dirname(os.path.dirname(__file__))
    eval_path = os.path.join(base, "ejercicio1-chainHumanEval", "evaluation.py")
    spec = importlib.util.spec_from_file_location("eh_eval", eval_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tdd_loop(task: Dict[str, Any], llm_tests: ModelWrapper, llm_code: ModelWrapper, max_attempts: int = MAX_ATTEMPTS) -> Dict[str, Any]:
    evaluation = _load_evaluation_module()
    history = []
    best = {"passed": -1, "code": None, "tests": None}
    spec = task["prompt"]
    for attempt in range(1, max_attempts + 1):
        print(f"--- Intento {attempt} ({task['task_id']}) ---")
        tests = llm_tests.generate_tests(spec)
        print("Tests generados:")
        for line in tests:
            print("  ", line)

        code = llm_code.generate_code(spec, tests)
        print("Código generado:\n", code)

        test_source = "\n".join(tests)
        # Construir tarea temporal para evaluate_task
        temp_task = {
            "prompt": task["prompt"],
            "entry_point": task["entry_point"],
            "test": test_source,
            "task_id": task["task_id"],
        }
        result = evaluation.evaluate_task(temp_task, code, timeout_seconds=10)

        passed = 1 if result.passed else 0
        errors = [result.error] if result.error else []

        print(f"Passed {passed}/{len(tests)}")

        state = {"attempt": attempt, "tests": tests, "code": code, "passed": passed, "errors": errors}
        history.append(state)

        if passed > best["passed"]:
            best.update({"passed": passed, "code": code, "tests": tests})

        if result.passed:
            print("Todos los tests pasaron. Solución encontrada.")
            break
        else:
            print("Fallo en tests; reintentando con feedback...\n")

    graph = {"nodes": [], "edges": []}
    prev_node = None
    for s in history:
        node = {"id": f"{task['task_id']}_attempt_{s['attempt']}", "passed": s["passed"]}
        graph["nodes"].append(node)
        if prev_node is not None:
            graph["edges"].append({"from": prev_node, "to": node["id"]})
        prev_node = node["id"]

    result = {"history": history, "best": best, "graph": graph}
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TDD con LangGraph + HumanEval")
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--tests-temperature", type=float, default=float(os.getenv("TEMPERATURE_TESTS", "0.6")))
    parser.add_argument("--code-temperature", type=float, default=float(os.getenv("TEMPERATURE_CODE", "0.0")))
    parser.add_argument("--max-attempts", type=int, default=int(os.getenv("MAX_ATTEMPTS", "5")))
    parser.add_argument("--tests-model-name", type=str, default=os.getenv("HUGGINGFACE_MODEL_TESTS", "mistralai/Mistral-7B-Instruct"))
    parser.add_argument("--code-model-name", type=str, default=os.getenv("HUGGINGFACE_MODEL_CODE", "bigcode/starcoder"))
    args = parser.parse_args()

    tests_temp = args.tests_temperature
    code_temp = args.code_temperature
    sample = None
    evaluation = _load_evaluation_module()
    sample = evaluation.load_humaneval_sample(args.sample_size, seed=42)

    # Crear wrappers HF con temperaturas distintas (razonamiento vs código)
    llm_tests = HFTestGenerator(args.tests_model_name, temperature=tests_temp)
    llm_code = HFCodeGenerator(args.code_model_name, temperature=code_temp)

    overall = {"tasks": []}
    out_dir = os.path.dirname(__file__)
    for task in sample:
        res = tdd_loop(task, llm_tests, llm_code, max_attempts=args.max_attempts)
        task_id_safe = task["task_id"].replace("/", "_")
        with open(os.path.join(out_dir, f"graph_{task_id_safe}.json"), "w") as f:
            json.dump(res["graph"], f, indent=2)
        with open(os.path.join(out_dir, f"history_{task_id_safe}.json"), "w") as f:
            json.dump(res["history"], f, indent=2)
        overall["tasks"].append({"task_id": task["task_id"], "best_passed": res["best"]["passed"]})

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("Procesado. Resultados guardados en el directorio del script.")


if __name__ == "__main__":
    main()
