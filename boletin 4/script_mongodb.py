import os
from pymongo import MongoClient
from smolagents import CodeAgent, tool, LiteLLMModel

# 1. Configuración del Modelo
model = LiteLLMModel(
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct", 
    api_key=os.getenv("HF_TOKEN")
)

# 2. Conexión y Autosiemsa de Datos (Simulando Chinook)
# Usamos 'mongo' como hostname si usas docker-compose, o localhost con --network=host
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client['tienda_musica']

def seed_data():
    """Crea datos de prueba similares a Chinook si la colección está vacía."""
    if "productos" not in db.list_collection_names():
        print("🌱 Sembrando datos de prueba en MongoDB...")
        db.productos.insert_many([
            {"nombre": "Guitarra Eléctrica", "precio": 599, "stock": 10, "categoria": "Instrumentos"},
            {"nombre": "Amplificador 50W", "precio": 199, "stock": 5, "categoria": "Audio"},
            {"nombre": "Púa de Litio", "precio": 2, "stock": 100, "categoria": "Accesorios"},
            {"nombre": "Cuerdas de Acero", "precio": 12, "stock": 50, "categoria": "Accesorios"}
        ])
        print("✅ Datos creados correctamente.")

# 3. Herramientas para el Agente
@tool
def list_collections() -> list:
    """Lists all available collections in the MongoDB database."""
    return db.list_collection_names()

@tool
def query_mongo(collection_name: str, query_filter: dict) -> str:
    """
    Queries a MongoDB collection using a filter dictionary.
    Example query_filter: {"categoria": "Audio"}
    """
    try:
        results = list(db[collection_name].find(query_filter).limit(10))
        for res in results: res['_id'] = str(res['_id']) # Limpieza de IDs
        return str(results)
    except Exception as e:
        return f"Error: {str(e)}"

# 4. Agente
agent = CodeAgent(tools=[list_collections, query_mongo], model=model, add_base_tools=False)

# 5. Interfaz Interactiva
if __name__ == "__main__":
    seed_data()
    print("\n🍃 Terminal MongoDB (Chinook Style) activa.")
    while True:
        try:
            prompt = input("👤 Tú (Mongo) > ")
            if prompt.lower() in ["salir", "exit"]: break
            agent.run(prompt)
        except Exception as e:
            print(f"⚠️ {e}")
