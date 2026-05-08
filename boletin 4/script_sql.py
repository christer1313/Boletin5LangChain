import sqlite3
import os
from smolagents import CodeAgent, LiteLLMModel, tool

# El token se recupera de la variable de entorno de Docker
hf_token = os.getenv("HF_TOKEN")

model = LiteLLMModel(
    # El prefijo 'huggingface/' le dice a LiteLLM que use la API de HF
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct", 
    api_key=os.getenv("HF_TOKEN"), # Tu token de HF
    temperature=0.2
)

DB_PATH = "Chinook.sqlite"

# 2. Herramientas
@tool
def get_schema() -> str:
    """
    Returns the database schema. Use this first to see tables and columns.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return "\n".join([table[0] for table in tables if table[0]])

@tool
def run_sql_query(query: str) -> str:
    """
    Executes a SELECT SQL query on the database.
    Args:
        query: A valid SQL SELECT statement.
    """
    if "SELECT" not in query.upper():
        return "Error: Only SELECT queries are allowed."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return str(conn.execute(query).fetchall())
    except Exception as e:
        return f"SQL Error: {str(e)}"

# 3. El Agente
agent = CodeAgent(
    tools=[get_schema, run_sql_query],
    model=model,
    add_base_tools=False
)

# Prueba del ejercicio
print(agent.run("Can you insert some invoices into the database to test this.Remeber the number. How many invoices were there in 2009 and 2011? Is it okay?"))
