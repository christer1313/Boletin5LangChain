"""
Lógica de evaluación automática de tareas HumanEval para el ejercicio 3.

Esta copia es independiente de la del ejercicio 1 para no compartir estado
ni dependencias entre carpetas.

Incluye:
- Carga del dataset
- Ejecución segura de código con timeout
- Evaluación de tareas individuales
- Selección aleatoria por defecto, con semilla opcional para reproducibilidad
"""

from __future__ import annotations

import multiprocessing as mp
import random
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


def load_humaneval_sample(sample_size: int, seed: int | None = None):
    """
    Descarga el dataset HumanEval de Hugging Face y selecciona una muestra aleatoria.

    - Intenta cargar el split 'test', luego 'validation', luego 'train'
    - Si ninguno existe, usa el primer split disponible
    - Baraja el dataset con una semilla opcional
    - Si seed es None, usa una semilla aleatoria distinta en cada ejecución

    Args:
        sample_size: Número de tareas a seleccionar.
        seed: Semilla opcional para reproducibilidad.

    Returns:
        Dataset con sample_size ejemplos.

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

    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)

    return dataset.shuffle(seed=seed).select(range(sample_size))


def _execute_tests(candidate_source: str, entry_point: str, test_source: str, queue: mp.Queue):
    """Ejecuta el código completado y sus tests en un namespace aislado."""
    namespace: dict[str, Any] = {}
    try:
        exec(candidate_source, namespace)
        candidate = namespace.get(entry_point)
        if candidate is None or not callable(candidate):
            raise RuntimeError(f"Entry point {entry_point!r} was not defined")
        namespace["candidate"] = candidate
        exec(test_source, namespace)
    except Exception as exc:  # noqa: BLE001 - report all failures uniformly
        queue.put({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    else:
        queue.put({"passed": True, "error": None})


def evaluate_task(task: dict[str, Any], completion: str, timeout_seconds: int) -> EvaluationResult:
    """Evalúa una tarea individual de HumanEval en un proceso aislado."""
    candidate_source = task["prompt"].rstrip() + "\n" + completion.lstrip()
    # En Linux, 'fork' evita que el hijo intente reimportar el módulo cargado
    # dinámicamente (por ejemplo, como 'eh_eval') y falle con ModuleNotFoundError.
    start_method = "fork" if mp.get_start_method(allow_none=True) == "fork" else "spawn"
    ctx = mp.get_context(start_method)
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
