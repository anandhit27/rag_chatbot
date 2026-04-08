from langchain_groq import ChatGroq
import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# ----------------------------
# Session state
# ----------------------------
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="RAG Internship Project")
st.title("📄 RAG Project (Groq + BM25)")

# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Save uploaded PDF
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")
    st.session_state.doc_loaded = True

    # ----------------------------
    # Load PDF
    # ----------------------------
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # ----------------------------
    # Split PDF into chunks
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # ----------------------------
    # BM25 Retriever
    # ----------------------------
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 4

    st.success("Document indexed successfully (BM25)")
    st.write(f"Total chunks created: {len(chunks)}")

    # ----------------------------
    # Delete / Reset option
    # ----------------------------
    if st.button("🗑️ Delete Document"):
        st.session_state.doc_loaded = False
        st.rerun()

    # ----------------------------
    # Ask question
    # ----------------------------
    query = st.text_input("ask a question from the document")
    if query:
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            st.info("No related content found in the uploaded document. (No answer provided)")
        else:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs]).strip()
            if len(context) < 50:
                st.info("No related content found in the uploaded document. (No answer provided)")
            else:
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0
                )
                prompt = f"""
Answer the question using only the information in the context below. If the answer is not present, say "I don't know,this is not related to document".

Context:
{context}

Question:
{query}
"""
                response = llm.invoke(prompt)
                st.subheader("✅ Answer")
                st.write(response.content)



   