# from dotenv import load_dotenv
# import os
#
# load_dotenv()
# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#
# import fitz  # For reading and extracting text from PDFs
# from sentence_transformers import SentenceTransformer  # For embedding text
# import chromadb  # For storing and querying vector embeddings
# from huggingface_hub import InferenceClient  # To access LLM from HuggingFace
#
# # 1. Load and chunk PDF into paragraphs
# doc = fitz.open("data/agentic_rag_paper.pdf")
# chunks = [b for page in doc for b in page.get_text().split("\n\n") if b.strip()]
#
# # 2. Create vector embeddings and store in ChromaDB
# model = SentenceTransformer("all-MiniLM-L6-v2")
# db = chromadb.Client().create_collection("agentic_rag")
# for i, c in enumerate(chunks):
#     db.add(documents=[c], ids=[str(i)], metadatas=[{"source": f"chunk_{i}"}], embeddings=[model.encode(c).tolist()])
#
# # 3. Chat loop: take question, retrieve relevant chunks, and generate answer
# llm = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)
#
# while True:
#     q = input("Ask a question (or 'exit'): ")
#     if q.lower() in ["exit", "quit"]:
#         break
#     retrieved = db.query(query_embeddings=[model.encode(q).tolist()], n_results=3)["documents"][0]
#     context = "\n".join(retrieved)
#     prompt = f"Answer based only on this research paper:\n{context}\n\nQ: {q}\nA:"
#     print("\n🧠", llm.text_generation(prompt, max_new_tokens=200))



from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

import fitz  # For reading and extracting text from PDFs
from sentence_transformers import SentenceTransformer  # For embedding text
import chromadb  # For storing and querying vector embeddings
from huggingface_hub import InferenceClient  # To access LLM from HuggingFace

# 1. Load and chunk PDF into paragraphs
doc = fitz.open("agentic_rag_paper.pdf")
chunks = [b for page in doc for b in page.get_text().split("\n\n") if b.strip()]

# 2. Create vector embeddings and store in ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client().create_collection("agentic_rag")

for i, c in enumerate(chunks):
    db.add(
        documents=[c],
        ids=[str(i)],
        metadatas=[{"source": f"chunk_{i}"}],
        embeddings=[model.encode(c).tolist()]
    )

# 3. Initialize the LLM client
llm = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# 4. Chat loop
while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break

    try:
        # Retrieve relevant chunks
        retrieved = db.query(
            query_embeddings=[model.encode(q).tolist()],
            n_results=3
        )["documents"][0]
        context = "\n".join(retrieved)

        # Construct better prompt
        prompt = f"Answer the following question using only this research paper:\n\n{context}\n\nQuestion: {q}\nAnswer:"

        # Get LLM response
        response = llm.text_generation(prompt, max_new_tokens=200)
        print("\n🧠", response.strip())

    except Exception as e:
        print(f"⚠️ Error: {e}")






