import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
from lxml import html
from tqdm import tqdm
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

# -------------------------------------------------------------
# ENVIRONMENT CONFIGURATION
# -------------------------------------------------------------
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dotenv_path = os.path.join(_project_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
load_dotenv(override=False)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Configure the page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Set up Gemini API
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if api_key:
        api_key = api_key.strip()
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
    return api_key

# Gemini model functions
def get_gemini_model(model_name=DEFAULT_GEMINI_MODEL, temperature=0.4):
    return genai.GenerativeModel(model_name)

def get_generation_config(temperature=0.4):
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
        raise e

# Prompt creation functions
def create_search_prompt(query, context=""):
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    Consider the provided context if available. 
    If the query requires external information, No context is provided, Irrelevant context is present or latest information is required, then output the special token <SEARCH> 
    followed by relevant keywords extracted from the query to optimize for search engine results. 
    Ensure the keywords are concise and relevant. If document data is sufficient, simply return blank."""
    
    if context:
        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"
    return f"{system_prompt}\n\nQuery: {query}"

def create_summary_prompt(content):
    return f"""Please provide a comprehensive yet concise summary of the following content, highlighting the most important points and maintaining factual accuracy. Organize the information in a clear and coherent manner:

Content to summarize:
{content}

Summary:"""

# Web scraping functions
def init_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def extract_static_page(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching page: {e}")
        return None
        
def extract_dynamic_page(url, driver):
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        body = driver.find_element(By.TAG_NAME, "body")
        ActionChains(driver).move_to_element(body).perform()
        time.sleep(random.uniform(2, 5))
        page_source = driver.page_source
        tree = html.fromstring(page_source)
        text = tree.xpath('//body//text()')
        text_content = ' '.join(text).strip()
        return text_content[:1000]
    except Exception as e:
        st.error(f"Error fetching dynamic page: {e}")
        return None

def scrape_page(url):
    if "javascript" in url or "dynamic" in url:
        driver = init_selenium_driver()
        text = extract_dynamic_page(url, driver)
        driver.quit()
    else:
        text = extract_static_page(url)
    return text

def scrape_web(urls, max_urls=5):
    texts = []
    for url in urls[:max_urls]:
        text = scrape_page(url)
        if text:
            texts.append(text)
        else:
            st.warning(f"Failed to retrieve content from {url}")
    return texts

def check_search_needed(model, query, context):
    prompt = create_search_prompt(query, context)
    try:
        response = generate_gemini_response(model, prompt)
    except Exception:
        return True, query
    
    if "<SEARCH>" in response:
        search_terms = response.split("<SEARCH>")[1].strip()
        return True, search_terms
    return False, None

def summarize_content(model, content):
    prompt = create_summary_prompt(content)
    return generate_gemini_response(model, prompt)

def process_query(query, context=''):
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
                "summary": f"Could not initialize search tool: {e}"
            }
        
        needs_search, search_terms = check_search_needed(model, query, context)
        
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
                    result["web_content"] = f"Error during search: {e}"
            
            with st.spinner("Summarizing search results..."):
                try:
                    summary = summarize_content(model, result["web_content"] or query)
                    result["summary"] = summary
                except Exception as e:
                    result["summary"] = f"Summarization failed: {e}\n\nSearch Content:\n{result['web_content']}"
        
    return result

# UI Components
def display_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.write("🔍")
    with col2:
        st.title("AI Research Assistant - Web Search")
    st.markdown("---")

def display_results(result):
    if result.get("needs_search"):
        st.success("Research completed!")
        with st.expander("Search Details", expanded=False):
            st.subheader("Search Terms Used")
            st.info(result.get("search_terms", "N/A"))
            st.subheader("Raw Web Content")
            st.text_area("Web Content", result.get("web_content", ""), height=200)
        st.subheader("Summary of Findings")
        st.markdown(result.get("summary", "No summary available."))
    else:
        st.info("Based on the analysis, no web search was needed for this query.")

def main():
    display_header()
    api_key = setup_gemini()
    
    with st.sidebar:
        st.subheader("API Status")
        if api_key:
            st.success("✅ Gemini API connected")
        else:
            st.error("❌ Gemini API key missing. Please set GEMINI_API_KEY in your `.env` file.")
            
        st.subheader("About")
        st.write("This AI Research Assistant uses Gemini and LangChain to search and summarize web content.")
        st.write("---")
        st.write("Powered by:")
        st.write("- Google Gemini API")
        st.write("- LangChain")
        st.write("- DuckDuckGo Search")
    
    with st.form("query_form"):
        query = st.text_area("Enter your research question", height=100, 
                           placeholder="E.g., What are the latest developments in quantum computing?")
        context = st.text_area("Optional: Add any context", height=100, 
                               placeholder="Add any additional context that might help with the research")
        submit_button = st.form_submit_button("🔍 Research")

    if submit_button and query:
        if not api_key:
            st.error("Please provide a valid GEMINI_API_KEY in your `.env` file.")
        else:
            result = process_query(query, context)
            display_results(result)

if __name__ == "__main__":
    main()
