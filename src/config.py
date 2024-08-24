import json
import os

with open("./secrets/openai_api_key.json", "r") as f:
    openai_api_key = json.load(f)
    print(openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key["OPENAI_API_KEY"]
    os.environ["OPENAI_ORGANIZATION"] = openai_api_key["OPENAI_ORGANIZATION"]

CHROMADB_HOST = "chromadb"
CHROMADB_PORT = 8000
