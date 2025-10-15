Bedrock RAG with Streamlit and ChromaDB

This project is a Retrieval-Augmented Generation (RAG) application built with Streamlit, AWS Bedrock, and ChromaDB.
It allows users to upload PDF documents, automatically process and embed their content into a vector database, and then query the documents conversationally using a chat interface powered by Claude Sonnet or Titan Embeddings.

🚀 Features

📤 PDF Upload – Upload local PDF documents directly from the Streamlit interface.

🧩 Chunking & Embedding – Documents are split into manageable text chunks and converted into embeddings using Amazon Titan (v1).

💾 Vector Storage – All embeddings and metadata are stored persistently in ChromaDB.

🧠 RAG Chat Interface – Ask natural language questions and receive answers generated with contextual retrieval from your uploaded documents.

🗑️ Document Management – Delete embeddings of specific documents or clear the database entirely.

💬 Persistent Chat History – Conversations remain across sessions, with an option to clear the chat history.

⚙️ AWS Bedrock Integration – Uses amazon.titan-embed-text-v1 for embeddings and claude-3-sonnet (with Haiku fallback) for text generation.

Clone the repo
```
git clone https://github.com/furkanfrost/bedrock-rag-first.git
cd bedrock-rag-first 
```

Install Dependencies
```
pip install -r requirements.txt
```

Run the app
```
streamlit run streamlit_app.py
```

🔐 AWS Access Requirements

This project uses Amazon Bedrock services for embedding and text generation.
Access to these models requires AWS SSO authentication with valid Bedrock permissions.

⚠️ Important:
The AWS SSO configuration and credentials used for this project are private and cannot be shared publicly.
Anyone who wants to run this application must:

Have access to an AWS account with Bedrock enabled in their region (e.g., eu-central-1).

Configure their own AWS SSO or IAM credentials locally using:
```
aws sso login --profile <your_profile_name>
```

Ensure they have these permissions:

`bedrock:InvokeModel`

`bedrock:InvokeModelWithResponseStream`

Without proper AWS SSO setup, the embedding and chat generation features will not work.
