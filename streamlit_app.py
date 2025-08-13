# streamlit_app.py
import os
import streamlit as st
from pypdf import PdfReader
import io


import chromadb
from chromadb import PersistentClient


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"

# RAG parameters
TOP_K = 5           # En yakın 5 chunk
CHUNK_SIZE = 1000   # Her chunk 1000 karakter
CHUNK_OVERLAP = 150 # Chunk'lar arasında 150 karakter overlap

st.set_page_config(page_title="Streamlit Bedrock RAG",  layout="wide")
st.title("RAG App")


# Chroma bağlantısını cacheleme(uygulama boyunca 1 kez oluşturulsun)
@st.cache_resource(show_spinner=False)
def get_chroma(path: str = CHROMA_PATH):
    os.makedirs(path, exist_ok=True)
    client = PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine mesafe
    )
    return client, collection


# Uygulama başında Chroma client/collection al
client, collection = get_chroma()


# --- Kenar çubuğu: ayarlar + (ileride dolacak) indekslenmiş belgeler listesi
with st.sidebar:

    st.subheader("Indexed Documents")

    # NOT: Şu an henüz upload/embed yapmadığımız için liste boş olabilir.
    # Demo listesi: Chroma'da kayıtlı metadatalardan benzersiz doc_hash → doc_name eşlemesi çıkarıyoruz.
    try:
        # Küçük setlerde pratik yaklaşım: ilk 500 kaydı alıp benzersiz doc'ları topla
        rows = collection.get(include=["metadatas"], limit=500)
        seen = {}
        for md in rows.get("metadatas", []):
            if not md:
                continue
            h = md.get("doc_hash")
            n = md.get("doc_name", h)
            if h and h not in seen:
                seen[h] = n

        if not seen:
            st.caption("No documents indexed yet.")
        else:
            for doc_hash, doc_name in seen.items():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**{doc_name}**")
                with c2:
                    if st.button("Delete", key=f"del_{doc_hash}"):
                        try:
                            collection.delete(where={"doc_hash": doc_hash})
                            st.toast(f"Deleted embeddings for {doc_name}")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
    except Exception as e:
        st.error(f"List error: {e}")

# --- Ana alan: PDF yükleme (henüz chunk/embedding yok)
st.subheader("📤 Upload PDF")
uploaded_pdfs = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_pdfs:
    for uf in uploaded_pdfs:
        try:
            raw = uf.read()
            reader = PdfReader(io.BytesIO(raw))
            page_count = len(reader.pages)
            st.success(f"Uploaded: {uf.name} — {page_count} pages")

            # Önizleme (ilk sayfa)
            if page_count > 0:
                first_page_text = reader.pages[0].extract_text() or ""
                if first_page_text.strip():
                    st.code(first_page_text[:500], language="markdown")
                else:
                    st.caption("No extractable text on page 1.")
        except Exception as e:
            st.error(f"Error reading {uf.name}: {e}")

# Chat skeleton (RAG bağlı değil)
st.caption("In the next step, the upload embed will be added.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("assistant"):
        st.write("In the next step, the upload embed will be added.")
