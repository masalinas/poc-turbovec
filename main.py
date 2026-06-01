import logging

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext 
from llama_index.llms.ollama import Ollama 
from llama_index.embeddings.ollama import OllamaEmbedding

from turbovec.llama_index import TurboQuantVectorStore
from turbovec import IdMapIndex 

logging.basicConfig(level=logging.DEBUG)

EMBEDDING_DIM = 768
#EMBEDDING_DIM = 1024

# Setup 011ama LLM and embeddings - fully local, nothing leaves your machine 
#Settings.llm = Ollama(model="hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest", request_timeout=120.0)
Settings.llm = Ollama(model="hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:latest", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:latest") 
#Settings.embed_model = OllamaEmbedding(model_name="hf.co/nadeem1362/mxbai-embed-large-v1-Q4_K_M-GGUF:latest") 

# Load document 
print ("Loading document...")
documents = SimpleDirectoryReader(input_files=["/home/miguel/temp/veradoc/Fusion_Models.pdf"]).load_data()

# Create turbovec vector store with 4-bit TurboOuant compression
print("Creating Turbovec vector index...")
tq_index = IdMapIndex(dim=EMBEDDING_DIM, bit_width=4) 
vector_store = TurboQuantVectorStore(index=tq_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index the document
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context)
print("Document indexed successfully with Turbovec compression.")

# Create query engine
query_engine = index.as_query_engine()

# Ask questions
questions = [
    "List authors",
    "Name of the IMU used",
]

print("--- Local RAG Pipeline: TurboQuant + Llama-3.2 + Ollama ---\n")
for q in questions:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")