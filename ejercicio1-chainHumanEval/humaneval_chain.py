"""
Script principal para automatizar la evaluación de HumanEval.

Orquesta el flujo completo:
1. Carga una muestra del dataset HumanEval
2. Construye la chain LLM para generar código
3. Para cada tarea: genera código, lo ejecuta, valida con tests
4. Reporta el porcentaje de aciertos
"""

from __future__ import annotations

import argparse

from chain import build_chain, build_model
from evaluation import evaluate_task, load_humaneval_sample, EvaluationResult


DEFAULT_SAMPLE_SIZE = 30
DEFAULT_TIMEOUT_SECONDS = 10


def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos del script.
    
    Disponibles:
    - --sample-size: Número de tareas a evaluar (default: 30)
    - --seed: Semilla para reproducibilidad (default: 42)
    - --temperature: Temperatura del modelo (default: 0.0, más determinista)
    - --timeout: Tiempo máximo por tarea en segundos (default: 10)
    
    Returns:
        Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Automated HumanEval evaluation with a LangChain chain")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> int:
    """
    Función principal que orquesta todo el flujo de evaluación.
    
    Proceso:
    1. Parsea argumentos de línea de comandos
    2. Carga una muestra de HumanEval
    3. Construye el modelo LLM y la chain
    4. Para cada tarea:
       - Invoca la chain para generar código
       - Evalúa el código con evaluate_task
       - Reporta resultado (OK/FAIL)
    5. Calcula y muestra el porcentaje de aciertos
    
    Returns:
        0 si ejecuta correctamente, puede lanzar excepciones en caso de error.
    """
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("sample-size must be at least 1")

    sample = load_humaneval_sample(args.sample_size, args.seed)
    model = build_model(args.temperature)
    chain = build_chain(model)

    results = []
    for index, task in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] {task['task_id']}")
        # Invocar el modelo con reintentos exponenciales
        max_attempts = 3
        backoff = 1.0
        completion = None
        for attempt in range(1, max_attempts + 1):
            try:
                completion = chain.invoke({"query": task["prompt"]})
                break
            except Exception as e:
                print(f"  warning: model call failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    print("  error: max retries reached; marking task as FAIL")
                else:
                    import time
                    time.sleep(backoff)
                    backoff *= 2
        if completion is None:
            results.append(EvaluationResult(task_id=task["task_id"], passed=False, error="ModelError"))
            print("  FAIL")
            print("  reason: ModelError")
            continue

        result = evaluate_task(task, completion, args.timeout)
        results.append(result)

        status = "OK" if result.passed else "FAIL"
        print(f"  {status}")
        if result.error:
            print(f"  reason: {result.error}")

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    accuracy = passed / total if total else 0.0

    print()
    print(f"Aciertos: {passed}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())