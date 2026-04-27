import streamlit as st
import requests
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.title("📚 AI Study Assistant")

# ---------------- API SETUP ----------------

from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def query_model(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",   # free + fast
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ---------------- EMBEDDING ----------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ---------------- PDF ----------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def split_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = split_text(text)

    embeddings = embedder.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    question = st.text_input("Ask a question")

    if question:
        query_vec = embedder.encode([question])
        distances, indices = index.search(query_vec, 3)

        context = " ".join([chunks[i] for i in indices[0]])

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

        answer = query_model(prompt)

        st.subheader("📌 Answer")
        st.write(answer)

        if st.button("🔁 Paraphrase"):
            para_prompt = f"Rewrite this simply:\n{answer}"
            para = query_model(para_prompt)

            st.subheader("📝 Paraphrased")
            st.write(para)