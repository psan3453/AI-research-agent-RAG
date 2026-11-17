import os
import time
import random
from typing import List, Optional

import streamlit as st
import numpy as np
import requests
import pdfplumber
import arxiv
import wikipedia
from bs4 import BeautifulSoup
from lxml import html
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
from langchain_community.tools import DuckDuckGoSearchRun
from pinecone import Pinecone, ServerlessSpec

# -------------------------------------------------------------
# ENVIRONMENT & CONFIGURATION
# -------------------------------------------------------------
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dotenv_path = os.path.join(_project_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
load_dotenv(override=False)

def get_config_var(name: str, fallback_name: Optional[str] = None) -> Optional[str]:
    """Retrieve an API key or configuration value from session state, environment, or Streamlit secrets."""
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

# Supported Groq Models
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
    """Retrieve available chat models for the Groq account or fallback to standard models."""
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

# Supported Gemini Models
SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# UTILITY FUNCTIONS & EMBEDDINGS
# -------------------------------------------------------------

class GeminiEmbeddings(Embeddings):
    """Gemini Embeddings class with multi-model fallback and 768-dim output."""
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
                    model=m,
                    content=text,
                    task_type=task_type,
                    output_dimensionality=768
                )
                return self._convert_to_float32(res["embedding"])
            except Exception:
                try:
                    res = genai.embed_content(
                        model=m,
                        content=text,
                        task_type=task_type
                    )
                    emb = self._convert_to_float32(res["embedding"])
                    if len(emb) > 768:
                        emb = emb[:768]
                    return emb
                except Exception:
                    continue
        raise RuntimeError("Failed to generate embedding with any available Gemini embedding model.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t, "retrieval_document") for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text, "retrieval_query")
    
    @staticmethod
    def _convert_to_float32(embedding):
        return np.array(embedding, dtype=np.float32).tolist()

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def read_data_from_doc(uploaded_file) -> List[Document]:
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

def make_chunks(docs: List[Document], chunk_len: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_len, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

def get_gemini_model(model_name: Optional[str] = None):
    chosen_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return genai.GenerativeModel(chosen_model)

def get_generation_config(temperature: float = 0.4):
    return {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

def get_safety_settings():
    return [
        {"category": category, "threshold": "BLOCK_NONE"}
        for category in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]

def generate_gemini_response(model, prompt: str) -> str:
    try:
        response = model.generate_content(
            prompt,
            generation_config=get_generation_config(),
            safety_settings=get_safety_settings()
        )
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
        return ''
    except Exception as e:
        err_str = str(e)
        if "404" in err_str or "not found" in err_str.lower():
            for fallback_name in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]:
                try:
                    fallback_model = genai.GenerativeModel(fallback_name)
                    response = fallback_model.generate_content(
                        prompt,
                        generation_config=get_generation_config(),
                        safety_settings=get_safety_settings()
                    )
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            return candidate.content.parts[0].text
                except Exception:
                    continue
        raise e

def summarize_text(text: str, gemini_key: Optional[str] = None) -> str:
    if not text:
        return "No text available to summarize."
    if not gemini_key:
        return text[:500] + ("..." if len(text) > 500 else "")
    try:
        genai.configure(api_key=gemini_key)
        model = get_gemini_model()
        prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"
        summary = generate_gemini_response(model, prompt_text)
        return summary if summary else text[:500] + "..."
    except Exception as e:
        return f"Summary unavailable: {str(e)[:100]}. Content snippet: {text[:300]}..."

# -------------------------------------------------------------
# RESEARCH ASSISTANT MODULE
# -------------------------------------------------------------

def download_pdf(pdf_url: str, save_path: str = "temp_paper.pdf") -> Optional[str]:
    try:
        response = requests.get(pdf_url, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return save_path
    except Exception as e:
        st.warning(f"Could not download paper PDF: {e}")
    return None

def search_arxiv(query: str, max_results: int = 2, gemini_key: Optional[str] = None) -> List[Document]:
    arxiv_docs = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        
        for result in client.results(search):
            pdf_link = next((link.href for link in result.links if 'pdf' in link.href), None)
            summary = "Summary not generated."
            
            if pdf_link:
                with st.spinner(f"Processing arXiv paper: {result.title}"):
                    pdf_path = download_pdf(pdf_link)
                    if pdf_path:
                        text = extract_text_from_pdf(pdf_path)
                        summary = summarize_text(text, gemini_key=gemini_key)
                        if os.path.exists(pdf_path):
                            try:
                                os.remove(pdf_path)
                            except OSError:
                                pass
                    else:
                        summary = "PDF could not be downloaded."
            else:
                summary = "No PDF available."

            content = f"""
            **Title:** {result.title}
            **Authors:** {', '.join(author.name for author in result.authors)}
            **Published:** {result.published.strftime('%Y-%m-%d')}
            **Abstract:** {result.summary}
            **PDF Summary:** {summary}
            **PDF Link:** {pdf_link if pdf_link else 'Not available'}
            """

            arxiv_docs.append(Document(page_content=content, metadata={"source": "arXiv", "title": result.title}))
    except Exception as e:
        st.warning(f"arXiv search encountered an error: {e}")
        
    return arxiv_docs

def search_wikipedia(query: str, max_results: int = 2) -> List[Document]:
    try:
        page_titles = wikipedia.search(query, results=max_results)
        wiki_docs = []
        for title in page_titles:
            try:
                with st.spinner(f"Processing Wikipedia article: {title}"):
                    page = wikipedia.page(title, auto_suggest=False)
                    wiki_docs.append(Document(
                        page_content=page.content[:2000], 
                        metadata={"source": "Wikipedia", "title": title}
                    ))
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                st.info(f"Skipping disambiguated or missing Wikipedia page '{title}': {e}")
            except Exception as e:
                st.info(f"Could not retrieve Wikipedia page '{title}': {e}")
        return wiki_docs
    except Exception as e:
        st.warning(f"Wikipedia search encountered an error: {e}")
        return []

class ResearchAssistant:
    def __init__(self, groq_api_key: Optional[str] = None, model: Optional[str] = None):
        self.groq_api_key = groq_api_key
        available = get_available_groq_models(groq_api_key)
        self.model_name = model or os.getenv("GROQ_MODEL") or (available[0] if available else "openai/gpt-oss-20b")
        self.llm = None
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Use the following context to answer the question. 
        If you don't know the answer, say so, but try your best to find relevant information 
        from the provided context and additional context.
        
        Context from user documents:
        {context}
        
        Additional context from research sources:
        {additional_context}
        
        Question: {input}
        
        Answer:
        """)
        
        if self.groq_api_key:
            try:
                self.llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model=self.model_name,
                    temperature=0.2
                )
                self.question_answer_chain = create_stuff_documents_chain(
                    self.llm, self.prompt
                )
            except Exception:
                self.llm = None

    def retrieve_documents(self, query: str, gemini_key: Optional[str] = None):
        user_context = []
        arxiv_docs = search_arxiv(query, gemini_key=gemini_key)
        wiki_docs = search_wikipedia(query)
        
        summarized_context = []
        for doc in arxiv_docs:
            summarized_context.append(f"**ArXiv - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content.strip()}...")
            
        for doc in wiki_docs:
            summarized_context.append(f"**Wikipedia - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content.strip()}...")
            
        return user_context, summarized_context
    
    def chat(self, question: str, groq_api_key: Optional[str] = None, gemini_key: Optional[str] = None):
        active_key = groq_api_key or self.groq_api_key
        if not active_key:
            return "❌ Groq API key is missing. Please set GROQ_API_KEY in your `.env` file or in the sidebar.", []

        if self.llm is None or active_key != self.groq_api_key:
            self.groq_api_key = active_key
            try:
                self.llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model=self.model_name,
                    temperature=0.2
                )
            except Exception as e:
                return f"❌ Failed to initialize Groq client: {str(e)}", []
        
        user_context, summarized_context = self.retrieve_documents(question, gemini_key=gemini_key)
        
        input_data = {
            "input": question,
            "context": "\n\n".join(user_context),
            "additional_context": "\n\n".join(summarized_context)
        }
        
        with st.spinner("Generating answer..."):
            prompt_text = f"""
            Question: {question}
            
            Additional context:
            {input_data['additional_context']}
            
            Please provide a comprehensive answer based on the above information.
            """
            try:
                response = self.llm.invoke(prompt_text)
                return response.content, summarized_context
            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "model_not_found" in err_msg:
                    # Automatic fallback to another available model
                    for fb_model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]:
                        if fb_model != self.model_name:
                            try:
                                fallback_llm = ChatGroq(api_key=self.groq_api_key, model=fb_model, temperature=0.2)
                                response = fallback_llm.invoke(prompt_text)
                                self.llm = fallback_llm
                                self.model_name = fb_model
                                return response.content, summarized_context
                            except Exception:
                                continue
                if "authentication" in err_msg.lower() or "api_key" in err_msg.lower():
                    return "❌ Groq Authentication Error: Invalid API key. Please check your GROQ_API_KEY.", summarized_context
                if "rate_limit" in err_msg.lower():
                    return "❌ Groq Rate Limit Reached: Please wait a moment before asking again.", summarized_context
                return f"❌ Groq API Error: {err_msg}", summarized_context

# -------------------------------------------------------------
# DOCUMENT QA MODULE
# -------------------------------------------------------------

def create_retrieval_chain_instance(uploaded_file, model_name: str, groq_key: str, gemini_key: str, pinecone_key: str):
    """Initializes embeddings, vector store, and retrieval chain for uploaded document."""
    if not groq_key:
        raise ValueError("Groq API key is required.")
    if not gemini_key:
        raise ValueError("Gemini API key is required for document embeddings.")
    if not pinecone_key:
        raise ValueError("Pinecone API key is required for vector storage.")

    # Configure embeddings
    embeddings = GeminiEmbeddings(api_key=gemini_key)
    
    # Read and process document
    docs = read_data_from_doc(uploaded_file)
    if not docs:
        raise ValueError("Could not extract any readable text from the uploaded PDF.")
    splits = make_chunks(docs)
    
    # Set up Pinecone vector store
    pc = Pinecone(api_key=pinecone_key)
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    index_name = "research-rag"
    
    if index_name not in index_names:
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
    
    llm = ChatGroq(model=model_name, temperature=0.75, api_key=groq_key)
    
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

# -------------------------------------------------------------
# WEB SEARCH MODULE
# -------------------------------------------------------------

def create_search_prompt(query: str, context: str = "") -> str:
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    Consider the provided context if available. 
    If the query requires external information, No context is provided, Irrelevant context is present, or latest information is required, then output the special token <SEARCH> 
    followed by relevant keywords extracted from the query to optimize for search engine results. 
    Ensure the keywords are concise and relevant. If document data is sufficient, simply return blank."""
    
    if context:
        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"
    return f"{system_prompt}\n\nQuery: {query}"

def create_summary_prompt(content: str) -> str:
    return f"""Please provide a comprehensive yet concise summary of the following content, highlighting the most important points and maintaining factual accuracy. Organize the information in a clear and coherent manner:

Content to summarize:
{content}

Summary:"""

def process_query(query: str, context: str = '', gemini_key: Optional[str] = None):
    if not gemini_key:
        return {
            "original_query": query,
            "needs_search": False,
            "search_terms": None,
            "web_content": None,
            "summary": "❌ Gemini API key is missing. Please set GEMINI_API_KEY in your `.env` file or sidebar."
        }

    genai.configure(api_key=gemini_key)
    with st.spinner("Processing query..."):
        model = get_gemini_model()
        try:
            search_tool = DuckDuckGoSearchRun()
        except Exception as e:
            return {
                "original_query": query,
                "needs_search": False,
                "search_terms": None,
                "web_content": None,
                "summary": f"❌ Search tool initialization error: {e}"
            }
        
        prompt = create_search_prompt(query, context)
        try:
            response = generate_gemini_response(model, prompt)
        except Exception:
            response = f"<SEARCH> {query}"
        
        needs_search = "<SEARCH>" in response
        search_terms = response.split("<SEARCH>")[1].strip() if needs_search else None
        
        result = {
            "original_query": query,
            "needs_search": needs_search,
            "search_terms": search_terms,
            "web_content": None,
            "summary": None
        }
        
        if needs_search and search_terms:
            with st.spinner(f"Searching the web for: {search_terms}"):
                try:
                    search_results = search_tool.run(search_terms)
                    result["web_content"] = search_results
                except Exception as e:
                    result["web_content"] = f"Web search could not retrieve results: {e}"
            
            with st.spinner("Summarizing search results..."):
                try:
                    summary_prompt = create_summary_prompt(result["web_content"] or query)
                    summary = generate_gemini_response(model, summary_prompt)
                    result["summary"] = summary
                except Exception as e:
                    result["summary"] = f"Summarization failed: {e}\n\nSearch output:\n{result['web_content']}"
        
    return result

# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------

def display_header():
    st.title("🔍 AI Research Assistant")
    st.markdown("Your all-in-one tool for research, document analysis, and web search")

def main():
    display_header()
    
    # Dynamic key retrieval
    groq_key = get_config_var("GROQ_API_KEY")
    pinecone_key = get_config_var("PINECONE_API_KEY")
    gemini_key = get_config_var("GEMINI_API_KEY", "GOOGLE_API_KEY")

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
        except Exception:
            pass

    groq_models = get_available_groq_models(groq_key)

    # Sidebar navigation & API status
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio("Choose a mode:", 
                          ["Research Assistant", "Document Q&A", "Web Search"])
        
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This AI Research Assistant helps you find and analyze information from various sources:
        - arXiv papers
        - Wikipedia articles
        - Your own uploaded PDF documents
        - Web search results
        """)
        
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
            
        if not (groq_key and gemini_key and pinecone_key):
            with st.expander("Configure Missing Keys"):
                if not groq_key:
                    user_groq = st.text_input("Enter GROQ_API_KEY", type="password", key="sidebar_groq")
                    if user_groq:
                        groq_key = user_groq.strip()
                        st.session_state["GROQ_API_KEY"] = groq_key
                        st.rerun()
                if not gemini_key:
                    user_gemini = st.text_input("Enter GEMINI_API_KEY", type="password", key="sidebar_gemini")
                    if user_gemini:
                        gemini_key = user_gemini.strip()
                        st.session_state["GEMINI_API_KEY"] = gemini_key
                        st.rerun()
                if not pinecone_key:
                    user_pinecone = st.text_input("Enter PINECONE_API_KEY", type="password", key="sidebar_pinecone")
                    if user_pinecone:
                        pinecone_key = user_pinecone.strip()
                        st.session_state["PINECONE_API_KEY"] = pinecone_key
                        st.rerun()

    # Research Assistant Mode
    if app_mode == "Research Assistant":
        st.header("Research Assistant")
        st.markdown("Ask research questions and get answers synthesized from arXiv papers and Wikipedia articles.")
        
        if not groq_key:
            st.warning("⚠️ Groq API key is not configured. Please add `GROQ_API_KEY` to your `.env` file or in the sidebar.")
        
        if "research_history" not in st.session_state:
            st.session_state.research_history = []
            
        if "research_assistant" not in st.session_state:
            st.session_state.research_assistant = ResearchAssistant(groq_api_key=groq_key)
        else:
            st.session_state.research_assistant.groq_api_key = groq_key

        with st.form(key="research_form"):
            question = st.text_input("Your research question:", key="research_question")
            submit_button = st.form_submit_button("Search")
            
        if st.button("Clear Chat"):
            st.session_state.research_history = []
            st.rerun()
        
        if submit_button and question:
            st.session_state.research_history.append({"role": "user", "content": question})
            answer, sources = st.session_state.research_assistant.chat(
                question, 
                groq_api_key=groq_key, 
                gemini_key=gemini_key
            )
            st.session_state.research_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
        
        for message in st.session_state.research_history:
            if message["role"] == "user":
                st.write(f"👤 **You:** {message['content']}")
            else:
                st.write(f"🤖 **AI Assistant:**")
                st.markdown(message["content"])
                
                if message.get("sources"):
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(source)
                            st.markdown("---")
    
    # Document Q&A Mode
    elif app_mode == "Document Q&A":
        st.header("Document Q&A")
        st.markdown("Upload a PDF document and ask questions about it using Pinecone vector retrieval and Groq LLM.")
        
        model_name = st.selectbox(
            "Select Groq Model",
            groq_models,
            index=0
        )
        
        if 'document_conversation' not in st.session_state:
            st.session_state.document_conversation = []
        
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file:
            if not (groq_key and gemini_key and pinecone_key):
                st.error("❌ Document Q&A requires GROQ_API_KEY, GEMINI_API_KEY (for embeddings), and PINECONE_API_KEY.")
            else:
                try:
                    with st.spinner("Processing document and generating vector embeddings..."):
                        chain = create_retrieval_chain_instance(
                            uploaded_file, 
                            model_name,
                            groq_key=groq_key,
                            gemini_key=gemini_key,
                            pinecone_key=pinecone_key
                        )
                    st.success("Document processed successfully! You can now ask questions.")
                    
                    for q, a in st.session_state.document_conversation:
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
                                    answer = result.get('answer', "No answer returned.")
                                    st.write(answer)
                                    st.session_state.document_conversation.append((question, answer))
                                except Exception as query_err:
                                    st.error(f"Error querying retrieval chain: {query_err}")
                        
                except Exception as e:
                    st.error(f"An error occurred while processing document: {str(e)}")
                    
        elif not (groq_key and gemini_key and pinecone_key):
            st.info("ℹ️ Note: Document Q&A requires Groq, Gemini (for embeddings), and Pinecone API keys configured.")
    
    # Web Search Mode
    else:
        st.header("Web Search")
        st.markdown("Search the web for up-to-date answers and synthesize summaries using Gemini.")
        
        if not gemini_key:
            st.warning("⚠️ Gemini API key is not configured. Please set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` or in the sidebar.")
        
        with st.form("web_query_form"):
            query = st.text_area("Enter your research question", height=100, 
                               placeholder="E.g., What are the latest developments in quantum computing?")
            context = st.text_area("Optional: Add any context", height=100, 
                                   placeholder="Add any additional context that might help with the research")
            submit_button = st.form_submit_button("🔍 Research")
        
        if submit_button and query:
            result = process_query(query, context, gemini_key=gemini_key)
            
            if result.get("needs_search"):
                st.success("Research completed!")
                
                with st.expander("Search Details", expanded=False):
                    st.subheader("Search Terms Used")
                    st.info(result.get("search_terms", "N/A"))
                    
                    st.subheader("Raw Web Content")
                    st.text_area("Web Content", result.get("web_content", ""), height=200)
                
                st.subheader("Summary of Findings")
                st.markdown(result.get("summary", "No summary available."))
            elif result.get("summary") and "❌" in result.get("summary"):
                st.error(result.get("summary"))
            else:
                st.info("Based on the analysis, no external web search was needed for this query.")

if __name__ == "__main__":
    main()
