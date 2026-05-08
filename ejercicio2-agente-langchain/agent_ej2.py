"""Agente del Ejercicio 2 (LangChain) con una sola chain expuesta como tool.

La idea es seguir el formato:

    @tool("nombre")
    def tool1(...):
        ...

    agent = create_agent(
        model=..., 
        tools=[tool1],
        system_prompt=...
    )

En este caso, la herramienta expuesta es `query_with_schema`, que orquesta la lógica completa:
1. Inferir la tabla relevante a partir de la query.
2. Obtener el esquema de esa tabla.
3. Generar la SQL con LLMChain + PromptTemplate.
4. Validar que sea una SELECT.
5. Ejecutar la consulta y devolver el resultado.
"""

import os
import re
import sqlite3

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.huggingface_hub import HuggingFaceHub

HF_TOKEN = os.getenv("HF_TOKEN")

# Base Chinook en el mismo nivel que este archivo.
CHINOOK_DB = os.path.normpath(os.path.join(os.path.dirname(__file__), "Chinook.sqlite"))


def _make_model() -> HuggingFaceHub:
    return HuggingFaceHub(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.2},
    )


def extract_code(text: str) -> tuple[str | None, str]:
    """Extract a fenced code block from a model response.

    Matches ```sql ... ``` or ``` ... ``` and returns (language, code).
    If no fenced block exists, returns (None, stripped_text).
    """
    match = re.search(r"```(?P<language>[^\n`]*)\n(?P<code>.*?)```", text, re.DOTALL)
    if not match:
        return None, text.strip()
    language = match.group("language").strip() or None
    code = match.group("code").strip()
    return language, code


def _execute_sql(query: str) -> str:
    """Helper: Execute a SQL query over the Chinook database and return the result as text.
    
    This is an internal helper, not exposed as a tool. It's used by query_with_schema
    for the final execution step (step 7 of the chain).
    """
    db = CHINOOK_DB
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(query)
        if cur.description:
            rows = cur.fetchall()
            headers = [d[0] for d in cur.description]
            out_lines = [", ".join(headers)]
            out_lines += [", ".join(map(str, row)) for row in rows]
            return "\n".join(out_lines)
        conn.commit()
        return "OK"
    except Exception as exc:
        return f"ERROR: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _list_tables() -> list:
    """Fetch all table names from Chinook database.

    This is an internal discovery helper used by the LLM to reason about which table
    is most relevant for the user's request.
    """
    try:
        conn = sqlite3.connect(CHINOOK_DB)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        rows = cur.fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_schema(table: str) -> str:
    """Get the schema (columns, types) for a given table using PRAGMA table_info."""
    try:
        conn = sqlite3.connect(CHINOOK_DB)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        rows = cur.fetchall()
        if not rows:
            return "<no schema available>"
        lines = ["cid | name | type | notnull | dflt_value | pk"]
        lines += [", ".join(map(str, r)) for r in rows]
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _generate_sql(schema: str, task: str, table: str) -> str:
    """Generate a SQL SELECT statement using LCEL pipe notation.

    This follows the pattern:
        prompt | model | StrOutputParser() | extract_code

    - PromptTemplate prepares the instruction.
    - The model generates a fenced code block containing SQL.
    - StrOutputParser normalizes the raw model output into text.
    - extract_code pulls the SQL out of the markdown fence.
    """
    prompt = PromptTemplate.from_template(
        "Given the table {table} with schema:\n{schema}\n\n"
        "Write a single valid SQL SELECT statement that accomplishes the task: {task}.\n"
        "Return the SQL inside a markdown code block and nothing else."
    )
    chain = prompt | _make_model() | StrOutputParser() | extract_code
    try:
        _, sql = chain.invoke({"schema": schema, "task": task, "table": table})
        return sql.strip()
    except Exception as exc:
        return f"ERROR_GENERATING_SQL: {exc}"


def _infer_table(task: str) -> str:
    """Infer the most relevant table using LCEL pipe notation.

    This mirrors the parser-style example with:
        prompt | model | StrOutputParser()

    The model is asked to return only the table name.
    """
    tables = _list_tables()
    if not tables:
        return ""

    prompt = PromptTemplate.from_template(
        "You are given a user request and the list of available SQLite tables.\n"
        "Choose the single most relevant table name for the request.\n\n"
        "User request: {task}\n\n"
        "Available tables:\n{tables}\n\n"
        "Return only one exact table name from the list. If you are unsure, return the best match."
    )
    chain = prompt | _make_model() | StrOutputParser()
    try:
        selected = chain.invoke({"task": task, "tables": "\n".join(tables)}).strip()
    except Exception:
        selected = ""

    if selected in tables:
        return selected

    # Fallback: if the model returns extra text, keep the first exact match it mentions.
    for table in tables:
        if table.lower() in selected.lower():
            return table

    return ""


@tool("query_with_schema")
def query_with_schema(task: str) -> str:
    """
    Chain orchestration for SQL generation and execution.
    
    This tool implements a multi-step workflow:
    1. Infer the most relevant table from the user request.
    2. Obtain the schema of that table.
    3. Use LLMChain to generate a safe SELECT statement from the schema + task description.
    4. Validate the generated SQL (must be SELECT).
    5. Execute the SQL and return results.
    
    This is a key example of LangChain modularity: each step is separated,
    the LLM is invoked only when needed (schema->SQL generation), and failures
    can be caught/validated before DB execution.
    
    Args:
        task (str): Natural language description of what information to retrieve.
        
    Returns:
        str: Formatted result including generated SQL and query output, or error message.
    """
    selected = _infer_table(task)
    if not selected:
        return "No se pudo inferir una tabla relevante a partir de la consulta."

    schema = _get_schema(selected)
    if schema.startswith("ERROR") or schema == "<no schema available>":
        return f"No se pudo obtener el esquema de la tabla '{selected}'.\n{schema}"

    sql = _generate_sql(schema, task, selected)
    if sql.startswith("ERROR"):
        return sql

    # Validación sencilla: obligar a SELECT
    if not sql.lower().lstrip().startswith("select"):
        return "La SQL generada no es una SELECT segura. Abortando.\nSQL generada:\n" + sql

    # Ejecutar la consulta usando el helper interno
    result = _execute_sql(sql)
    return f"Tabla inferida: {selected}\n\nSQL ejecutada:\n{sql}\n\nResultado:\n{result}"


agent = create_agent(
    model=_make_model(),
    tools=[query_with_schema],
    system_prompt=(
        "You are a SQL assistant. Use the query_with_schema tool to infer the relevant table, "
        "inspect its schema, generate a safe SELECT statement, and return the result. "
        "Be helpful and concise in your explanations."
    ),
)


def run_agent(prompt: str) -> str:
    """Execute the agent and extract the last assistant response in a robust way."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])

        messages = result.get("messages")
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", last_message))
            return str(getattr(last_message, "content", last_message))

    return str(result)


def repl() -> None:
    print("REPL interactivo — escribe 'exit', 'quit' o 'salir' para terminar")
    try:
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "salir"}:
                print("Saliendo del REPL...")
                break

            try:
                print(run_agent(user_input))
            except Exception as exc:
                print(f"ERROR en ejecución del agente: {exc}")
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario. Saliendo...")


if __name__ == "__main__":
    repl()
