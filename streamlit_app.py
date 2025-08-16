# aws sso login --profile furkan

import os, io, uuid, hashlib, json
import streamlit as st
from pypdf import PdfReader
import chromadb
from chromadb import PersistentClient

# =========================
# values
# =========================
CHROMA_PATH = "chroma_db"
import os
print("CHROMA_PATH absolute:", os.path.abspath(CHROMA_PATH))

COLLECTION_NAME = "rag_collection"

TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# AWS Bedrock
AWS_PROFILE = os.getenv("AWS_PROFILE", "furkan")  # senin SSO profil adın
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")

# =========================
# functions
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def read_pdf(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def clean_spaces(s: str) -> str:
    return " ".join((s or "").split())

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = clean_spaces(text)
    step = max(1, size - max(0, overlap))
    chunks = []
    for i in range(0, len(text), step):
        ch = text[i:i+size]
        if ch:
            chunks.append(ch)
    return chunks

@st.cache_resource(show_spinner=False)
def get_chroma(path: str = CHROMA_PATH):
    os.makedirs(path, exist_ok=True)
    client = PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection

@st.cache_resource(show_spinner=False)
def get_bedrock_runtime(region: str = AWS_REGION, profile: str = AWS_PROFILE):
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime")

def embed_texts(runtime, model_id: str, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i, t in enumerate(texts):
        payload = {"inputText": t}
        try:
            resp = runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json",
            )
            body = json.loads(resp["body"].read())
            vec = body.get("embedding") or body.get("vector")
            if not vec:
                raise RuntimeError(f"Empty embedding at chunk {i}")
            vectors.append(vec)
        except Exception as e:
            preview = (t[:80] + "...") if len(t) > 80 else t
            raise RuntimeError(f"Embedding failed at chunk {i} (preview: {preview}): {e}") from e
    return vectors

# =========================
#app
# =========================
st.set_page_config(page_title="Streamlit Bedrock RAG", layout="wide")
st.title("RAG App")

client, collection = get_chroma()
bedrock_rt = get_bedrock_runtime()

# =========================
# Sidebar: Indexed Documents
# =========================
with st.sidebar:
    st.write("Collection name:", collection.name)
    st.write("DB count (sidebar):", collection.count())

with st.sidebar:
    st.subheader("Indexed Documents")

    # 1) Chroma'dan TÜM kayıtları sayfalayarak oku (limit 1000 ile eksik kalmasın)
    seen = {}
    try:
        total = collection.count()
        st.write(f"📦 Total embeddings in DB: {total}")

        batch = 500
        offset = 0
        while True:
            rows = collection.get(include=["metadatas"], limit=batch, offset=offset)
            ids = rows.get("ids") or []
            mds = rows.get("metadatas") or []
            for md in mds:
                if not md:
                    continue
                h = md.get("doc_hash")
                n = md.get("doc_name") or h
                if h and h not in seen:
                    seen[h] = n
            if len(ids) < batch:
                break
            offset += batch

    except Exception as e:
        st.error(f"List error: {e}")

    # 2) İşlemden hemen sonra görünmesi için session manifest'i de birleştir
    for h, n in (st.session_state.get("_last_indexed_docs") or {}).items():
        if h not in seen:
            seen[h] = n

    # 3) Listeyi göster + silme
    if not seen:
        st.caption("No documents indexed yet.")
    else:
        st.write(f"📄 Documents: {len(seen)}")
        for doc_hash, doc_name in seen.items():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{doc_name}**")
            with c2:
                if st.button("Delete", key=f"del_{doc_hash}"):
                    try:
                        collection.delete(where={"doc_hash": doc_hash})
                        # Manifest'ten de düş
                        if "_last_indexed_docs" in st.session_state:
                            st.session_state["_last_indexed_docs"].pop(doc_hash, None)
                        st.toast(f"Deleted embeddings for {doc_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

# =========================
# PDF Upload + Chunk + Embed + Store
# =========================
st.subheader(" Upload PDF (Chunk + Embed + Store)")
uploaded_pdfs = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs", type="primary") and uploaded_pdfs:
    with st.spinner("Processing PDFs..."):
        for uf in uploaded_pdfs:
            try:
                raw = uf.read()
                doc_name = uf.name
                doc_hash = sha256_bytes(raw)

                # Aynı belge eklenmiş mi kontrol
                existing = collection.get(where={"doc_hash": doc_hash}, limit=1)
                if existing.get("ids"):
                    st.info(f"Already indexed: {doc_name} (skipping)")
                    continue

                # Metin çıkarma
                text = read_pdf(raw)
                if not clean_spaces(text):
                    st.warning(f"No extractable text in {doc_name}")
                    continue

                # Chunk
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

                # Embed
                st.write(f" Embedding {len(chunks)} chunks with {EMBED_MODEL_ID}...")
                vecs = embed_texts(bedrock_rt, EMBED_MODEL_ID, chunks)
                st.write(f" Received {len(vecs)} embeddings.")

                # Chroma'ya yaz
                ids = [str(uuid.uuid4()) for _ in chunks]
                metas = [{"doc_name": doc_name, "doc_hash": doc_hash, "chunk_index": i} for i in range(len(chunks))]
                collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
                st.write("After add count:", collection.count())
                st.write("lens:", len(ids), len(chunks), len(metas), len(vecs))

                # 👉 1) (varsa) persist çağrısı - bazı sürümlerde gerekmez ama zararı yok
                if hasattr(client, "persist"):
                    try:
                        client.persist()
                    except Exception:
                        pass

                # 👉 2) Sidebar'da daha app tazelenmeden belge adını göstermek için manifest'e yaz
                st.session_state.setdefault("_last_indexed_docs", {})[doc_hash] = doc_name

                # 👉 3) İsteğe bağlı: anlık sayacı göster (debug)
                st.write("✅ After add, total in DB:", collection.count())

                st.success(f"Indexed {doc_name} → {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Failed on {uf.name}: {e}")
#    st.rerun()


# Chat skeleton

st.caption("Next step: retrieval + Claude (RAG).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("assistant"):
        st.write("RAG coming next. ✅")
