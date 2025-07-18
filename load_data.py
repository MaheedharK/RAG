from pymongo import MongoClient
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import openai
from langchain_community.chains import PebbloRetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param
import requests
import json

client = MongoClient(key_param.MONGO_URI)
dbName = "Rag_model"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

loader = DirectoryLoader('./sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

from typing import List

class LangChainCompatibleOllamaEmbeddings:
    def __init__(self, model_name="llama3"):
        self.base_url = "http://localhost:11434/api/embeddings"
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            response = requests.post(self.base_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                embeddings.append(data.get("embedding", []))
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        response = requests.post(self.base_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("embedding", [])
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


embeddings = LangChainCompatibleOllamaEmbeddings(model_name="llama3")

vectorStore = MongoDBAtlasVectorSearch.from_documents(
    documents=data,
    embedding=embeddings,
    collection=collection
)

