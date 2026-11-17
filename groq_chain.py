import os
from typing import List, Optional

import streamlit as st
import numpy as np
import pdfplumber
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone, ServerlessSpec

# -------------------------------------------------------------
# ENVIRONMENT CONFIGURATION
# -------------------------------------------------------------
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dotenv_path = os.path.join(_project_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
load_dotenv(override=False)

def get_config_var(name: str, fallback_name: Optional[str] = None) -> Optional[str]:
    val = st.session_state.get(name)
    if not val and fallback_name:
        val = st.session_state.get(fallback_name)
    if not val:
        val = os.getenv(name)
    if not val and fallback_name:
        val = os.getenv(fallback_name)
    if not val:
        try:
            if hasattr(st, "secrets"):
                val = st.secrets.get(name)
                if not val and fallback_name:
                    val = st.secrets.get(fallback_name)
        except Exception:
            pass
    if val and isinstance(val, str):
        val = val.strip()
    return val if val else None

FALLBACK_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
]

def get_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    if not api_key:
        return FALLBACK_GROQ_MODELS
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.models.list()
        chat_models = [
            m.id for m in resp.data 
            if not m.id.startswith("whisper") and "guard" not in m.id and "orpheus" not in m.id
        ]
        if chat_models:
            for fb in FALLBACK_GROQ_MODELS:
                if fb not in chat_models:
                    chat_models.append(fb)
            return chat_models
    except Exception:
        pass
    return FALLBACK_GROQ_MODELS

# Set page configuration
st.set_page_config(page_title="Document Q&A with Groq", layout="wide")

groq_key = get_config_var("GROQ_API_KEY")
pinecone_key = get_config_var("PINECONE_API_KEY")
gemini_key = get_config_var("GEMINI_API_KEY", "GOOGLE_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
    except Exception:
        pass

groq_models = get_available_groq_models(groq_key)

# Sidebar
with st.sidebar:
    st.title("Model Selection")
    model_name = st.selectbox(
        "Select Groq Model",
        groq_models,
        index=0
    )
    st.markdown("---")
    st.subheader("API Status")
    if groq_key:
        st.success("✅ Groq API connected")
    else:
        st.error("❌ Groq API key missing")
    if gemini_key:
        st.success("✅ Gemini API connected")
    else:
        st.error("❌ Gemini API key missing")
    if pinecone_key:
        st.success("✅ Pinecone API connected")
    else:
        st.error("❌ Pinecone API key missing")

# Main content
st.title("RAG-powered Document Assistant")
st.write("Upload a document and ask questions about it!")

# Gemini Embeddings implementation
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        if not api_key:
            raise ValueError("Gemini API key is required for embeddings.")
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
    
    def _embed_single(self, text: str, task_type: str) -> List[float]:
        candidates = [self.model_name, "models/gemini-embedding-001", "models/text-embedding-004", "models/embedding-001"]
        for m in candidates:
            try:
                res = genai.embed_content(
                    model=m, content=text, task_type=task_type, output_dimensionality=768
                )
                return self._convert_to_float32(res["embedding"])
            except Exception:
                try:
                    res = genai.embed_content(model=m, content=text, task_type=task_type)
                    emb = self._convert_to_float32(res["embedding"])
                    if len(emb) > 768:
                        emb = emb[:768]
                    return emb
                except Exception:
                    continue
        raise RuntimeError("Failed to generate embedding with Gemini embedding models.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t, "retrieval_document") for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text, "retrieval_query")
    
    @staticmethod
    def _convert_to_float32(embedding):
        return np.array(embedding, dtype=np.float32).tolist()

def read_data_from_doc(uploaded_file):
    docs = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = "\n".join([
                "\n".join(["\t".join(cell if cell is not None else "" for cell in row) for row in table])
                for table in tables if table
            ]) if tables else ""
            images = page.images
            image_text = f"[{len(images)} image(s) detected]" if images else ""
            content = f"{text}\n\n{table_text}\n\n{image_text}".strip()
            if content:
                docs.append(Document(page_content=content, metadata={"page": i + 1}))
    return docs

def make_chunks(docs, chunk_len=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_len, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

def get_retrieval_chain(uploaded_file, model, groq_key_val, gemini_key_val, pinecone_key_val):
    with st.spinner("Processing document... This may take a minute."):
        embeddings = GeminiEmbeddings(api_key=gemini_key_val)
        docs = read_data_from_doc(uploaded_file)
        if not docs:
            raise ValueError("No text could be extracted from the document.")
        splits = make_chunks(docs)
        
        # Set up vector store
        pc = Pinecone(api_key=pinecone_key_val)
        index_name = "research-rag"
        indexes = pc.list_indexes()
        if index_name not in [idx.name for idx in indexes]:
            try:
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            except Exception:
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=768,
                        metric="cosine"
                    )
                except Exception:
                    pass
                    
        vectorstore = PineconeVectorStore.from_documents(
            splits,
            embeddings,
            index_name=index_name,
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        llm = ChatGroq(model=model, temperature=0.75, api_key=groq_key_val)
        
        system_prompt = """
        You are an AI assistant answering questions based on retrieved documents and additional context. 
        Use the provided context from both database retrieval and additional sources to answer the question. 

        - **Discard irrelevant context:** If one of the contexts (retrieved or additional) does not match the question, ignore it.
        - **Highlight conflicting information:** If multiple sources provide conflicting information, explicitly mention it.
        - **Prioritize accuracy:** If neither context provides a relevant answer, say "I don't know" instead of guessing.

        Provide concise yet informative answers, ensuring clarity and completeness.

        Retrieved Context: {context}
        Additional Context: {additional_context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}\n\nRetrieved Context: {context}\n\nAdditional Context: {additional_context}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    if not (groq_key and gemini_key and pinecone_key):
        st.error("Please configure all required API keys (GROQ_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY) in your `.env` file.")
    else:
        try:
            chain = get_retrieval_chain(
                uploaded_file, 
                model_name,
                groq_key_val=groq_key,
                gemini_key_val=gemini_key,
                pinecone_key_val=pinecone_key
            )
            
            st.success("Document processed successfully! You can now ask questions.")
            
            for q, a in st.session_state['conversation']:
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)
            
            question = st.chat_input("Ask a question about your document...")
            
            if question:
                with st.chat_message("user"):
                    st.write(question)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            result = chain.invoke({
                                "input": question,
                                "additional_context": ""
                            })
                            answer = result.get('answer', "No answer generated.")
                            st.write(answer)
                            st.session_state['conversation'].append((question, answer))
                        except Exception as invoke_err:
                            st.error(f"Error during query answering: {invoke_err}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
elif not (groq_key and gemini_key and pinecone_key):
    st.info("Please make sure all required API keys are configured in your `.env` file.")
