import subprocess
import os
from smolagents import CodeAgent, tool, LiteLLMModel

# El token se recupera de la variable de entorno de Docker
hf_token = os.getenv("HF_TOKEN")

model = LiteLLMModel(
    # El prefijo 'huggingface/' le dice a LiteLLM que use la API de HF
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct", 
    api_key=os.getenv("HF_TOKEN"), # Tu token de HF
    temperature=0.2
)

@tool
def execute_bash(command: str) -> str:
    """
    Executes a bash command and returns the output.
    Args:
        command: The command to run (e.g. 'ls', 'mkdir data').
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)

agent = CodeAgent(tools=[execute_bash], model=model, add_base_tools=True)

# Ejemplo de entrada del ejercicio:
print(agent.run("Write a python script that takes two numbers and multiply them, save it as mult.py and then run it."))