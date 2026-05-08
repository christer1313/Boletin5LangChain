"""
Lógica de evaluación automática de tareas HumanEval.

Este módulo contiene:
- Carga del dataset
- Ejecución segura de código con timeout
- Evaluación de tareas individuales
- Orquestación de benchmarks completos
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


DATASET_NAME = "openai/openai_humaneval"


@dataclass(slots=True)
class EvaluationResult:
    """Resultado de la evaluación de una tarea individual."""
    task_id: str
    passed: bool
    error: str | None


def load_humaneval_sample(sample_size: int, seed: int):
    """
    Descarga el dataset HumanEval de Hugging Face y selecciona una muestra aleatoria.
    
    - Intenta cargar el split 'test', luego 'validation', luego 'train'
    - Si ninguno existe, usa el primer split disponible
    - Baraja el dataset con una semilla para reproducibilidad
    - Selecciona los primeros N elementos
    
    Args:
        sample_size: Número de tareas a seleccionar (mínimo 30 recomendado).
        seed: Semilla para reproducibilidad del shuffle.
        
    Returns:
        Dataset con sample_size ejemplos. Cada tarea (task) contiene:
        - task_id: Identificador único (ej: "HumanEval/0")
        - prompt: Descripción y cabecera de la función a completar (ej: "def has_close_elements(...)")
        - entry_point: Nombre de la función (ej: "has_close_elements")
        - test: Código con los asserts de validación
        - canonical_solution: Solución de referencia (para consulta)
        
    Raises:
        ValueError: Si sample_size es mayor que el tamaño del dataset.
    """
    dataset_dict = load_dataset(DATASET_NAME)
    for split_name in ("test", "validation", "train"):
        if split_name in dataset_dict:
            dataset = dataset_dict[split_name]
            break
    else:
        dataset = next(iter(dataset_dict.values()))

    if sample_size > len(dataset):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size {len(dataset)}")

    return dataset.shuffle(seed=seed).select(range(sample_size))


def _execute_tests(candidate_source: str, entry_point: str, test_source: str, queue: mp.Queue):
    """
    Ejecuta el código completado y sus tests en un namespace aislado.
    
    Esta función se usa internamente en un proceso separado para:
    1. Ejecutar el código completado (prompt + modelo)
    2. Verificar que la función de entrada existe
    3. Ejecutar los tests (asserts) contra esa función
    4. Reportar éxito o fallo a través de la queue
    
    Args:
        candidate_source: El código completo (prompt + completación).
        entry_point: Nombre de la función a evaluar (ej: "has_close_elements").
        test_source: Código con los asserts que validan la función.
        queue: Queue multiprocessing para comunicar resultado al proceso padre.
    """
    namespace: dict[str, Any] = {}
    try:
        exec(candidate_source, namespace)
        candidate = namespace.get(entry_point)
        if candidate is None or not callable(candidate):
            raise RuntimeError(f"Entry point {entry_point!r} was not defined")
        namespace["candidate"] = candidate
        exec(test_source, namespace)
    except Exception as exc:  # noqa: BLE001 - we want to report any failure uniformly
        queue.put({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    else:
        queue.put({"passed": True, "error": None})


def evaluate_task(task: dict[str, Any], completion: str, timeout_seconds: int) -> EvaluationResult:
    """
    Evalúa una tarea individual de HumanEval en un proceso aislado.
    
    Proceso:
    1. Combina el prompt original con la completación del modelo
    2. Inicia un proceso hijo para ejecutar y testear
    3. Espera a que termine (con timeout)
    4. Si excede timeout, termina el proceso y reporta fallo
    5. Captura el resultado desde la queue multiprocessing
    
    Args:
        task: Diccionario con los campos de HumanEval (prompt, entry_point, test, task_id).
        completion: Código generado por el LLM.
        timeout_seconds: Tiempo máximo de ejecución antes de matar el proceso.
        
    Returns:
        EvaluationResult con:
        - task_id: Identificador de la tarea (ej: "HumanEval/0")
        - passed: True si todos los tests pasaron, False en caso contrario
        - error: Descripción del error (None si pasó todos los tests)
    """
    candidate_source = task["prompt"].rstrip() + "\n" + completion.lstrip()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_tests,
        args=(candidate_source, task["entry_point"], task["test"], queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return EvaluationResult(task_id=task["task_id"], passed=False, error="Timeout")

    if queue.empty():
        return EvaluationResult(task_id=task["task_id"], passed=False, error="No result returned")

    result = queue.get()
    return EvaluationResult(task_id=task["task_id"], passed=result["passed"], error=result["error"])
