# aws sso login --profile furkan

import os, io, uuid, hashlib, json, textwrap
import streamlit as st
from pypdf import PdfReader

import chromadb
from chromadb import PersistentClient

# mutlak path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_collection"

# RAG parametreleri (sabit)
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# AWS / Bedrock
AWS_PROFILE = os.getenv("AWS_PROFILE", "furkan")
AWS_REGION  = os.getenv("AWS_REGION",  "eu-central-1")

# Embedding (Titan v1, 1536-dim)
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")

# Claude Sonnet (Bedrock)
CLAUDE_MODEL_ID = os.getenv("BEDROCK_CLAUDE_MODEL_ID", "eu.anthropic.claude-3-7-sonnet-20250219-v1:0")

# System prompts
SYSTEM_PROMPT = textwrap.dedent("""
You are a careful RAG assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.
 Be concise and factual.eu.anthropic.claude-3-7-sonnet-20250219-v1:0
""").strip()

st.set_page_config(page_title="Streamlit Bedrock RAG", layout="wide")
st.title("RAG App")

# Chat history permanent
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(messages: list[dict]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"History save failed: {e}")

def clear_history_file():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        st.error(f"History clear failed: {e}")


# funcs

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
    """Titan v1 ile embed; list[str] -> list[vector]"""
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
            preview = (t[:120] + "…") if len(t) > 120 else t
            raise RuntimeError(f"Embedding failed at chunk {i} (preview: {preview}): {e}") from e
    return vectors

def call_claude(runtime, model_id: str, system_prompt: str, user_text: str) -> str:
    """Bedrock üzerinden Claude Sonnet çağrısı (Anthropic format)"""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.2,
        "system": [{"type": "text", "text": system_prompt}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        ],
    }
    resp = runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())

    parts = data.get("content", [])
    text = "".join([p.get("text", "") for p in parts if p.get("type") == "text"])
    return text.strip()


client, collection = get_chroma()
bedrock_rt = get_bedrock_runtime()

# Sidebar: Indexed Documents

with st.sidebar:
    st.subheader("Indexed Documents")

    # embedding count
    try:
        total = collection.count()
        st.write(f"📦 Total embeddings in DB: {total}")
    except Exception as e:
        st.error(f"Count error: {e}")
        total = 0

    # doc_hash → doc_name
    seen = {}
    try:
        batch, offset = 500, 0
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
                        st.toast(f"Deleted embeddings for {doc_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    st.divider()
    st.subheader("Chat Controls")
    if st.button("🧹 Clear chat history"):
        st.session_state["messages"] = []
        clear_history_file()
        st.success("Chat history cleared.")
        st.rerun()

# PDF Upload,chunk,embed,store

st.subheader("📤 Upload PDF ")
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

                # skip same docs
                existing = collection.get(where={"doc_hash": doc_hash}, limit=1)
                if existing.get("ids"):
                    st.info(f"Already indexed: {doc_name} (skipping)")
                    continue

                # chunk → embed
                text = read_pdf(raw)
                if not clean_spaces(text):
                    st.warning(f"No extractable text in {doc_name}")
                    continue

                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                vecs = embed_texts(bedrock_rt, EMBED_MODEL_ID, chunks)

                # write chroma
                ids = [str(uuid.uuid4()) for _ in chunks]
                metas = [{"doc_name": doc_name, "doc_hash": doc_hash, "chunk_index": i} for i in range(len(chunks))]
                collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)

                st.success(f"Indexed {doc_name} → {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Failed on {uf.name}: {e}")
    st.rerun()


#(RAG)

st.divider()
st.subheader("💬 Ask something")

if "messages" not in st.session_state:
    st.session_state.messages = load_history()


# 1) history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 2) input
question = st.chat_input("Type your question about the uploaded PDFs...")

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})
    save_history(st.session_state.messages)

    try:
        q_vec = embed_texts(bedrock_rt, EMBED_MODEL_ID, [question])[0]
        res = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        mds  = (res.get("metadatas") or [[]])[0]
        dists= (res.get("distances") or [[]])[0]

        if docs:
            parts = []
            for i, (t, md) in enumerate(zip(docs, mds), start=1):
                name = md.get("doc_name", "unknown")
                idx  = md.get("chunk_index", i-1)
                snippet = (t or "").strip()
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + " ..."
                parts.append(f"[{i}] Document: {name} | chunk #{idx}\n{snippet}")
            ctx_text = "\n\n".join(parts)
            final_user = f"QUESTION:\n{question}\n\nCONTEXT (top-{TOP_K}):\n{ctx_text}"
        else:
            final_user = f"QUESTION:\n{question}\n\nCONTEXT:\n(Empty — if answer is not known from memory, say you don't know.)"

        # 3) Claude
        try:
            answer = call_claude(bedrock_rt, CLAUDE_MODEL_ID, SYSTEM_PROMPT, final_user)
        except Exception as e:

            answer = f"Claude call failed: {e}"

    except Exception as e:
        answer = f"RAG failed: {e}"
        docs, mds, dists = [], [], []

    #  assistant message
    with st.chat_message("assistant"):
        st.markdown(answer or "_(empty answer)_")
        with st.expander("🔎 Used context"):
            if not docs:
                st.caption("No retrieved context.")
            else:
                for i, (t, md, dist) in enumerate(zip(docs, mds, dists), start=1):
                    name = md.get("doc_name", "unknown")
                    idx  = md.get("chunk_index", "-")
                    st.markdown(f"**[{i}] {name} — chunk #{idx} — distance: {dist:.4f}**")
                    st.code((t or "")[:800], language="markdown")


    st.session_state.messages.append({"role": "assistant", "content": answer or ""})
    save_history(st.session_state.messages)

