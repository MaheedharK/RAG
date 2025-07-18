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
dbName = "Emergency_Aid"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

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

vectorStore = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection
)

from langchain_core.language_models.llms import LLM
from typing import List
import requests

class LocalOllamaLLM(LLM):
    def __init__(self, model_name="llama3", url="http://localhost:11434/api/chat", temperature=0):
        self.model_name = model_name
        self.url = url
        self.temperature = temperature

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        response = requests.post(self.url, json=payload)

        if response.status_code == 200:
            data = response.json()
            return data["message"]["content"]
        else:
            raise Exception(f"LLM call failed: {response.status_code} {response.text}")

    @property
    def _llm_type(self) -> str:
        return "local_ollama"

def query_data(query): 
    docs = vectorStore.similarity_search(query, k=1)
    as_output = docs[0].page_content

    llm = LocalOllamaLLM(model_name="llama3", temperature=0)
    retriever = vectorStore.as_retriever()
    qa = PebbloRetrievalQA._chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output

with gr.Blocks(theme=Base(), title="Twitter Verification Bot") as demo:
    gr.Markdown(
        """
        # verifying the model
        """)
    textbox = gr.Textbox(label="Enter Your Questions:")
    with gr.Row():
        button = gr.Button("submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Output with Embedded documents ")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Output along with Ollama and Vector Search")

    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()
