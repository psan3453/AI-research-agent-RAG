import os
import requests
import pdfplumber
import arxiv
import wikipedia
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional

# Load environment variables from .env in project root or current working directory
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dotenv_path = os.path.join(_project_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
load_dotenv(override=False)

def get_config_var(name: str, fallback_name: Optional[str] = None) -> Optional[str]:
    val = st.session_state.get(name) if "st" in globals() and hasattr(st, "session_state") else None
    if not val and fallback_name and "st" in globals() and hasattr(st, "session_state"):
        val = st.session_state.get(fallback_name)
    if not val:
        val = os.getenv(name)
    if not val and fallback_name:
        val = os.getenv(fallback_name)
    if val and isinstance(val, str):
        val = val.strip()
    return val if val else None

# API Keys
GROQ_API_KEY = get_config_var('GROQ_API_KEY')
GEMINI_API_KEY = get_config_var('GEMINI_API_KEY', 'GOOGLE_API_KEY')
PINECONE_API_KEY = get_config_var('PINECONE_API_KEY')
PINECONE_ENV = get_config_var('PINECONE_ENV') or 'research-rag'

# Supported Models
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Configure Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

def get_generation_config():
    return {
        'temperature': 0.4,
        'top_p': 0.95,
        'top_k': 40,
        'max_output_tokens': 4096,
    }

def get_safety_settings():
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

def get_gemini_model(model_name=DEFAULT_GEMINI_MODEL, temperature=0.4):
    return genai.GenerativeModel(model_name)

def generate_gemini_response(model, prompt):
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
            try:
                fallback_model = genai.GenerativeModel("gemini-1.5-flash")
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
                pass
        return ''

def download_pdf(pdf_url, save_path="temp_paper.pdf"):
    try:
        response = requests.get(pdf_url, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return save_path
    except Exception as e:
        st.warning(f"Could not download PDF: {e}")
    return None

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        st.warning(f"Error extracting text from PDF: {e}")
        return ""

def summarize_text(text, gemini_key=None):
    active_key = gemini_key or GEMINI_API_KEY
    if not text or not active_key:
        return text[:500] + ("..." if len(text) > 500 else "")
    try:
        genai.configure(api_key=active_key)
        model = get_gemini_model()
        prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"
        summary = generate_gemini_response(model, prompt_text)
        return summary if summary else text[:500] + "..."
    except Exception:
        return text[:500] + "..."

def search_arxiv(query, max_results=2, gemini_key=None):
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
        st.warning(f"Error searching arXiv: {e}")
    return arxiv_docs

def search_wikipedia(query, max_results=2):
    try:
        page_titles = wikipedia.search(query, results=max_results)
        wiki_docs = []
        for title in page_titles:
            try:
                with st.spinner(f"Retrieving Wikipedia article: {title}"):
                    page = wikipedia.page(title, auto_suggest=False)
                    wiki_docs.append(Document(
                        page_content=page.content[:2000], 
                        metadata={"source": "Wikipedia", "title": title}
                    ))
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                st.info(f"Skipping disambiguated or missing Wikipedia page {title}: {e}")
            except Exception as e:
                st.info(f"Could not retrieve Wikipedia article {title}: {e}")
        return wiki_docs
    except Exception as e:
        st.warning(f"Error searching Wikipedia: {e}")
        return []

class ResearchAssistant:
    def __init__(self, groq_api_key=None, gemini_api_key=None, pinecone_api_key=None, model=DEFAULT_GROQ_MODEL):
        self.groq_api_key = groq_api_key or GROQ_API_KEY
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        self.pinecone_api_key = pinecone_api_key or PINECONE_API_KEY
        self.model_name = model
        self.vector_store = None
        self.retriever = None
        
        # Optional Pinecone vector store initialization
        if self.pinecone_api_key:
            index_name = "research-assistant"
            class SimpleEmbeddings(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0] * 1536 for _ in texts]
                def embed_query(self, text: str) -> List[float]:
                    return [0.0] * 1536
            
            try:
                pc = Pinecone(api_key=self.pinecone_api_key)
                indexes = pc.list_indexes()
                if index_name in [idx.name for idx in indexes]:
                    self.vector_store = PineconeVectorStore(
                        index_name=index_name, 
                        embedding=SimpleEmbeddings()
                    )
                    self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                self.vector_store = None
                self.retriever = None
        
        # Initialize LLM
        self.llm = None
        if self.groq_api_key:
            try:
                self.llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model=self.model_name,
                    temperature=0.2
                )
            except Exception:
                self.llm = None
        
        # Set up prompt template
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
        
        if self.llm:
            self.question_answer_chain = create_stuff_documents_chain(
                self.llm, self.prompt
            )
            if self.retriever:
                self.chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
            else:
                self.chain = None
        else:
            self.question_answer_chain = None
            self.chain = None

    def retrieve_documents(self, query):
        user_context = []
        
        if self.retriever:
            try:
                retrieved_docs = self.retriever.get_relevant_documents(query)
                for doc in retrieved_docs:
                    user_context.append(f"**{doc.metadata.get('source', 'Unknown Source')}**: {doc.page_content}...")
            except Exception:
                pass
        
        with st.spinner("Searching arXiv papers..."):
            arxiv_docs = search_arxiv(query, gemini_key=self.gemini_api_key)
        with st.spinner("Searching Wikipedia articles..."):
            wiki_docs = search_wikipedia(query)
        
        summarized_context = []
        for doc in arxiv_docs:
            summarized_context.append(f"**ArXiv - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
        for doc in wiki_docs:
            summarized_context.append(f"**Wikipedia - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
        return user_context, summarized_context
    
    def chat(self, question):
        if not self.groq_api_key:
            return "❌ Groq API key is missing. Please set GROQ_API_KEY in `.env`.", []
        
        if self.llm is None:
            try:
                self.llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model=self.model_name,
                    temperature=0.2
                )
                self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            except Exception as e:
                return f"❌ Failed to initialize Groq LLM: {e}", []

        user_context, summarized_context = self.retrieve_documents(question)
        
        input_data = {
            "input": question,
            "context": "\n\n".join(user_context),
            "additional_context": "\n\n".join(summarized_context)
        }
        
        with st.spinner("Generating answer..."):
            try:
                if self.chain:
                    result = self.chain.invoke(input_data)
                    return result["answer"], summarized_context
                else:
                    prompt_text = f"""
                    Question: {question}
                    
                    Additional context:
                    {input_data['additional_context']}
                    
                    Please provide a comprehensive answer based on the above information.
                    """
                    response = self.llm.invoke(prompt_text)
                    return response.content, summarized_context
            except Exception as e:
                err_msg = str(e)
                if "authentication" in err_msg.lower() or "api_key" in err_msg.lower():
                    return "❌ Invalid Groq API key. Please check your credentials.", summarized_context
                return f"❌ Error querying Groq: {err_msg}", summarized_context

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 AI Research Assistant")
    st.markdown("Ask me any research question, and I'll search for relevant information.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "assistant" not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
            
    with st.form(key="query_form"):
        question = st.text_input("Your research question:", key="question_input")
        submit_button = st.form_submit_button("Search")
        
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        answer, sources = st.session_state.assistant.chat(question)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"👤 **You:** {message['content']}")
        else:
            st.write(f"🤖 **AI Assistant:**")
            st.markdown(message["content"])
            
            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(source[:500] + "..." if len(source) > 500 else source)
                        st.markdown("---")
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI Research Assistant helps you find and analyze information from various sources:
        - arXiv papers
        - Wikipedia articles
        """)
        
        st.header("API Status")
        if GROQ_API_KEY:
            st.success("✅ Groq API connected")
        else:
            st.error("❌ Groq API key missing")
        if GEMINI_API_KEY:
            st.success("✅ Gemini API connected")
        else:
            st.info("ℹ️ Gemini API key missing (used for arXiv paper summaries)")

if __name__ == "__main__":
    main()
