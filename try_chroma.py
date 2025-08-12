import os, uuid
import chromadb
from chromadb import PersistentClient

# 1) Kalıcı depoyu hazırlayın
os.makedirs("chroma_db", exist_ok=True)
client = PersistentClient(path="chroma_db")

# 2) Koleksiyon oluşturun/varsa alın (cosine mesafe)
col = client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"},
)

# 3) Örnek belge ekleyin (embedding HENÜZ yok; sadece doküman + metadata)
doc_name = "dummy.txt"
doc_hash = "hash_dummy_001"  # ileride gerçek dosya hash'i kullanacağız
chunk_text = "Hello Chroma! This is a minimal test chunk."
col.add(
    ids=[str(uuid.uuid4())],
    documents=[chunk_text],
    metadatas=[{"doc_name": doc_name, "doc_hash": doc_hash, "chunk_index": 0}],
)

# 4) Metadata ile geri okuyun (query_texts yerine GET kullanıyoruz, çünkü embedding henüz yok)
rows = col.get(
    where={"doc_hash": doc_hash},
    include=["documents", "metadatas"],
)
print("Inserted count:", len(rows.get("ids", [])))
print("First doc text:", rows["documents"][0])

# 5) Silme testini yapın (belge bazlı silme)
col.delete(where={"doc_hash": doc_hash})
rows_after = col.get(where={"doc_hash": doc_hash})
print("After delete, count:", len(rows_after.get("ids", [])))

print("OK")
