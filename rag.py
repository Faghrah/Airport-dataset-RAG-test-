import os
from typing import List

import pandas as pd
import requests
from dotenv import dotenv_values
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_token: str):
        self.model_name = model_name
        self.api_token = api_token
        self.api_url = (
            "https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": texts,
                "options": {
                    "wait_for_model": True
                }
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face embedding request failed: "
                f"{response.status_code} {response.text}"
            )

        embeddings = response.json()

        if not isinstance(embeddings, list):
            raise RuntimeError(
                f"Unexpected embedding response: {embeddings}"
            )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


local_config = dotenv_values("config.txt")


def get_config_value(
    name: str,
    default: str | None = None
) -> str | None:
    return os.getenv(name) or local_config.get(name) or default


GROQ_API_KEY = get_config_value("GROQ_API_KEY")
HF_TOKEN = get_config_value("HF_TOKEN")
MONGODB_URI = get_config_value("MONGODB_URI")
DB_NAME = get_config_value("DB_NAME")
COLLECTION_NAME = get_config_value("COLLECTION_NAME")

GROQ_MODEL = get_config_value(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

EMBEDDING_MODEL = get_config_value(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_INDEX_NAME = get_config_value(
    "VECTOR_INDEX_NAME",
    "vector_index"
)


if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing")

if not DB_NAME:
    raise RuntimeError("DB_NAME is missing")

if not COLLECTION_NAME:
    raise RuntimeError("COLLECTION_NAME is missing")


os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]

embeddings = HuggingFaceAPIEmbeddings(
    model_name=EMBEDDING_MODEL,
    api_token=HF_TOKEN
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME
)

model = ChatGroq(
    model=GROQ_MODEL,
    temperature=0
)

def create_vector_index():
    vector_store.create_vector_search_index(
        dimensions=384,
        wait_until_complete=60
    )
    return {
        "message": "Vector index created successfully",
        "index_name": VECTOR_INDEX_NAME,
        "dimensions": 384
    }
def load_excel_to_database(file_path: str):
    df = pd.read_excel(file_path)
    if "airport_AR" in df.columns:
        df = df.drop(columns=["airport_AR"])
    required_columns = {"airport_EN", "year", "movement"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise RuntimeError(f"Missing columns: {missing_columns}")
    documents = []
    for index, row in df.iterrows():
        airport = str(row["airport_EN"]).strip()
        year = int(row["year"])
        movement = int(row["movement"])
        text = (
            f"Airport: {airport}. "
            f"Year: {year}. "
            f"Aircraft movement: {movement}."
        )
        document = Document(
            page_content=text,
            metadata={
                "airport": airport,
                "year": year,
                "movement": movement,
                "source": file_path,
                "row_number": int(index)
            }
        )
        documents.append(document)
    vector_store.add_documents(documents)
    return {
        "message": "Excel data added to MongoDB vector database",
        "rows_added": len(documents)
    }

def search_rag(question: str):
    results = vector_store.similarity_search_with_score(question,k=3)
    retrieved_chunks = []
    for doc, score in results:
        retrieved_chunks.append({
            "score": score,
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks
    }

def ask_rag(question: str):
    search_result = search_rag(question)
    chunks = search_result["retrieved_chunks"]
    context = ""
    for chunk in chunks:
        context += chunk["text"] + "\n"

    prompt = f"""
You are an airport movement data assistant.

Use only the context below to answer the user's question.
If the answer is not found in the context, say:
"I do not have enough information in the database."

Context:
{context}

Question:
{question}
"""
    response = model.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "retrieved_chunks": chunks
    }
def list_chunks():
    chunks = collection.find(
        {},
        {
            "_id": 0,
            "text": 1,
            "metadata": 1
        }
    ).limit(10)
    return list(chunks)
def clear_database():
    result = collection.delete_many({})
    return {
        "message": "Database cleared",
        "deleted_documents": result.deleted_count
    }