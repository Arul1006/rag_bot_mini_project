# 🧠 RAG Chatbot for Agentic RAG Paper

This is a Retrieval-Augmented Generation (RAG) chatbot that answers questions using a research paper on **Agentic RAG**. It uses a combination of sentence embeddings, vector search, and a Hugging Face large language model to generate context-aware answers.

## 📄 Overview

The chatbot:
1. Loads and splits a research paper (`agentic_rag_paper.pdf`) into chunks.
2. Converts the text chunks into vector embeddings using `all-MiniLM-L6-v2`.
3. Stores embeddings in a local ChromaDB vector store.
4. Retrieves the most relevant chunks based on the user's question.
5. Constructs a prompt and queries the `Mixtral-8x7B-Instruct-v0.1` model via Hugging Face.
6. Displays a response based only on the content of the paper.

## 🧰 Libraries & Frameworks Used

| Library             | Purpose |
|---------------------|---------|
| [`fitz` (PyMuPDF)]  | Extracts text from PDFs |
| [`sentence-transformers`] | Converts text chunks into embeddings |
| [`chromadb`]        | Vector database for storing/querying embeddings |
| [`huggingface_hub`] | Interfaces with hosted language models |
| [`dotenv`]          | Loads Hugging Face API token from `.env` securely |


## 🚀 Getting Started

### 1. Clone the Repository

- git clone https://github.com/your-username/rag-chatbot.git
- cd rag-chatbot


### 2. Set Up Virtual Environment

1. python -m venv .venv
2. source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. pip install -r requirements.txt
