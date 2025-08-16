import chromadb
from chromadb import PersistentClient

client = PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("rag_collection")

print("Count before:", collection.count())

collection.add(
    ids=["test1"],
    documents=["This is a test document."],
    metadatas=[{"doc_name": "debug.pdf", "doc_hash": "debug123", "chunk_index": 0}],
    embeddings=[[0.1]*1536],   # Titan v1 boyutunda sahte vektör
)

print("Count after:", collection.count())
