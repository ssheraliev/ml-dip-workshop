import streamlit as st
import os
import logging
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama # Correct import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_community.tools.tavily_search import TavilySearchResults

# --- st.set_page_config - Streamlit command ---
st.set_page_config(page_title="Python/ML Style Assistant", layout="wide")

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.info(f"Set TOKENIZERS_PARALLELISM to {os.environ.get('TOKENIZERS_PARALLELISM', 'Not Set')}")

# --- Web Search, Secrets, Embedding config ---
VECTORSTORE_PATH = "./chroma_db_nomic_v1_notebook_final"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
LLM_MODEL = "gemma3"
try:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    logger.info("TAVILY_API_KEY loaded from st.secrets.")
except KeyError:
    logger.warning("TAVILY_API_KEY not found in st.secrets or .streamlit/secrets.toml. Web search would not be availbale.")
    TAVILY_API_KEY = None
except FileNotFoundError:
    logger.warning(".streamlit/secrets.toml not found. Web search would not be availbale.")
    TAVILY_API_KEY = None


# --- web search helper ---
def format_tavily_results(results):
    """Tavily results formatting into a readable string."""
    if not results:
        return "No relevant information found over web search."
    try:
        if not isinstance(results, list): return "Received non-list results from Tavily."
        summary = "\n\n".join(
            f"URL: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}"
            for res in results if isinstance(res, dict)
        )
        return summary if summary else "Found results, but could not extract content."
    except Exception as e:
        logger.error(f"Error formatting Tavily results: {e}")
        return "Error processing search results."

# --- Components loading ---
@st.cache_resource
def load_components():
    """Components loading; embeddings, vectorstore, LLM, retriever."""
    logger.info("Attempting to load resources for Streamlit app...")
    embeddings_app = None
    retriever_app = None
    llm_app = None
    router_llm_app = None
    web_search_tool_app = None

    try:
        # Embeddings
        logger.info(f"Loading Nomic embeddings ({EMBEDDING_MODEL}, local)...")
        embeddings_app = NomicEmbeddings(model=EMBEDDING_MODEL, inference_mode="local")
        logger.info("Embeddings loaded.")

        # Vector store loading
        logger.info(f"Loading Chroma vector store from {VECTORSTORE_PATH}...")
        if not os.path.exists(VECTORSTORE_PATH):
            st.error(f"Vector store not found at {VECTORSTORE_PATH}. Please ensure it was built.")
            logger.error(f"Vector store directory not found: {VECTORSTORE_PATH}")
            return None, None, None, None, None
        vectorstore_app = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings_app)
        retriever_app = vectorstore_app.as_retriever(search_kwargs={"k": 3})
        logger.info("Vector store and retriever loaded.")

        # LLM init
        logger.info(f"Initializing Ollama LLM ({LLM_MODEL})...")
        llm_app = Ollama(model=LLM_MODEL, temperature=0)
        try:
             llm_app.invoke("Respond briefly: OK", config={'max_new_tokens': 5})
             logger.info("Ollama LLM connection verified.")
        except Exception as ollama_err:
             st.error(f"Failed to connect to Ollama ({LLM_MODEL}): {ollama_err}")
             logger.error(f"Ollama connection/invocation test failed: {ollama_err}")
             return embeddings_app, retriever_app, None, None, None
        router_llm_app = llm_app

        # Tavily init
        if TAVILY_API_KEY:
            logger.info("Initializing Tavily Search tool...")
            try:
                 web_search_tool_app = TavilySearchResults(k=3)
                 logger.info("Tavily search tool initialized.")
            except Exception as e:
                 st.warning(f"Failed to initialize Tavily even with API key: {e}. Proceeding without web search.")
                 logger.error(f"Tavily initialization failed: {e}")
                 web_search_tool_app = None
        else:
             logger.info("Tavily isn't initialized.")
             web_search_tool_app = None

        logger.info("All available resources loaded successfully for Streamlit.")
        return embeddings_app, retriever_app, llm_app, router_llm_app, web_search_tool_app

    except Exception as e:
        st.error(f"Error during resource loading: {e}")
        logger.exception(f"Error loading resources for Streamlit: {e}")
        return None, None, None, None, None


# components loading on app init phase
loaded_data = load_components()
if loaded_data is not None:
     embeddings_app, retriever_app, llm_app, router_llm_app, web_search_tool_app = loaded_data
     components_loaded_app = all([retriever_app, llm_app])
     if not components_loaded_app:
         logger.error("Essential components (retriever or LLM) failed to load.")
else:
     components_loaded_app = False
     embeddings_app, retriever_app, llm_app, router_llm_app, web_search_tool_app = None, None, None, None, None
     logger.error("Component loading failed critically.")


# --- main chains ---
full_chain_app = None

if components_loaded_app:
    logger.info("Defining Langchain chains for Streamlit app...")
    try:
        # Router chain
        router_prompt_template = """Given the user query, determine if it is best answered using internal knowledge on Python related tasks (use PEP8, Google Python Style Guide) and ML best practices (Google's Rules of ML), or if it requires a real-time web search for general programming topics, specific code examples not related to style/ML rules, current events, or very recent information.
Respond only with the word 'vectorstore' or 'web_search'.

User Query: {question}
Decision:"""
        router_prompt = PromptTemplate.from_template(router_prompt_template)
        router_app = (
            {"question": RunnablePassthrough()}
            | router_prompt
            | router_llm_app
            | StrOutputParser()
            | RunnableLambda(lambda x: x.strip().lower().replace("'", "").replace('"', ''))
        )

        # RAG chain
        rag_prompt_template = """You are coding assistant specializing in Python (use PEP8, Google Python Style Guide) and Machine Learning best practices (Google's Rules of ML). Answer the user's question based *only* on the following provided context. If the context doesn't contain the answer, state that the specific information isn't available in the provided knowledge base. Do not use external knowledge.

Context:
{context}

Question: {question}

Answer:"""
        rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
        rag_chain_app = (
            RunnablePassthrough.assign(
                context= RunnablePassthrough.assign(question=lambda x: x["question"])
                         | (lambda x: x["question"])
                         | retriever_app
                         | (lambda docs: "\n\n".join(doc.page_content for doc in docs))
            )
            | rag_prompt
            | llm_app
            | StrOutputParser()
        )


        # Web Search chain
        if web_search_tool_app:
            web_search_prompt_template = """You are a helpful Python coding assistant. Analyze the user's question and the provided web search results.

1.  **Determine Intent:** Is the user asking for an explanation, or for code implementation (e.g., "How to write...", "Implement...", "Code for...")?
2.  **Synthesize Answer:**
    *   If the user wants an explanation, provide a clear summary based on the context.
    *   If the user wants code: Generate clean, functional Python code based on the information in the web search results and the user's request.
3.  **Apply Style:** When generating Python code, make a best effort to adhere to the Google Python Style Guide (e.g., clear variable names, docstrings for functions/classes, reasonable line lengths, comments where necessary).
4.  **Handle Insufficient Info:** If the search results are irrelevant or insufficient to fulfill the request (either explanation or code), clearly state that.

Web Search Results:
{context}

Question: {question}

Answer:"""
            web_search_prompt = ChatPromptTemplate.from_template(web_search_prompt_template)
            web_chain_app = (
                RunnablePassthrough.assign(
                    context= RunnablePassthrough.assign(question=lambda x: x["question"])
                             | (lambda x: x["question"])
                             | web_search_tool_app
                             | RunnableLambda(format_tavily_results)
                )
                | web_search_prompt
                | llm_app
                | StrOutputParser()
            )
        else:
            web_chain_app = RunnableLambda(lambda x: "Web search is disabled - missing Tavily API key.")


        # Branching
        def decide_chain_app(route_info):
            """Which chain to run on router output"""
            question = route_info.get("question", "N/A")
            route_decision = route_info.get("route_decision", "").lower()
            logger.debug(f"App deciding chain for '{question[:50]}...'. Router decision: '{route_decision}'")

            if "vectorstore" in route_decision:
                logger.info(f"App: routing to RAG chain for question: {question[:50]}...")
                return rag_chain_app
            elif web_search_tool_app:
                logger.info(f"App: routing to Web Search chain for question: {question[:50]}...")
                return web_chain_app
            else:
                logger.warning(f"App: routing defaulted away from web search for prompt: {question[:50]}...")
                return web_chain_app


        routing_chain_app = RunnablePassthrough.assign(
             route_decision = router_app
        )

        full_chain_app = routing_chain_app | RunnableLambda(decide_chain_app)
        logger.info("Langchain chains defined successfully for Streamlit app.")

    except Exception as e:
        logger.exception("Error defining Langchain chains for Streamlit app.")
        st.error(f"Error setting up processing chains: {e}")
        full_chain_app = None
else:
    logger.error("Streamlit: chains cannot be defined as essential components failed to load.")


# --- Streamlit application ---
st.title("🥷 Python and Machine Learning tasks assistance")
st.caption(f"Running via Ollama ({LLM_MODEL}), Vector Database, Nomic Embeddings & Tavily")

# chat history init
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me about Python, Machine Learning, or general programming questions."}]

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if components_loaded_app and full_chain_app is not None:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                response = full_chain_app.invoke({"question": prompt})
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred during processing: {e}"
                logger.exception(f"App exception during chain invocation for prompt '{prompt}': {e}")
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"I've encountered an error: {e}"})
    else:
        # when components won't load
        st.error("Application components could not be loaded. Cannot process the request.")
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != "Error: Application components failed to load.":
             st.session_state.messages.append({"role": "assistant", "content": "Error: Application components failed to load."})