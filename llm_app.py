import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os

# LLM configuration
OLLAMA_MODEL_NAME = "local_llm"

# Langchain configuration
# -- Prompt template ---
template = """
Answer the question below based on the conversation history and the latest context

Here is the conversation history:
{context}

Question: {prompt}
"""
prompt = ChatPromptTemplate.from_template(template)

# API call
try:
    model = OllamaLLM(model=OLLAMA_MODEL_NAME)

except Exception as e:
    # error handling during init
    st.error(f"Failed to initialize Ollama '{OLLAMA_MODEL_NAME}'. Error: {e}")
    st.stop()

# Chain
chain = prompt | model

# Streamlit Application logic
st.title("🥷 Machine Learning and Python tasks assistance")
st.caption(f"Retrieval Augmented Generation as backbone ({OLLAMA_MODEL_NAME})")

# History and context string in session
if "message" not in st.session_state:
    st.session_state.message = []
if "context_string" not in st.session_state:
    # store cumulative context for LLM prompt
    st.session_state.context_string = ""

# existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["context"])

# user input using Streamlit's chat input
user_input = st.chat_input("Ask Machine Learning and Python problem questions...")

if user_input:
    # user message in graphical UI
    with st.chat_message("Engineer"):
        st.markdown(user_input)
    # user message to display history
    st.session_state.messages.append({"role": "Engineer", "context": user_input})

    # context and invoke the LangChain chain
    current_context = st.session_state.context_string
    try:
        # spinner while waiting for the response
        with st.spinner("Thinking..."):
            result = chain.invoke({"context": current_context, "question": user_input})

        # assistant response
        with st.chat_message("assistant"):
            st.markdown(result)
        # assistant response history on display
        st.session_state.messages.append({"role": "Assistant", "context": result})

        # append both use prompt and assistant's response
        st.session_state.context_string += f"\nEngineer: {user_input}\nAssistant: {result}"

    except Exception as e:
        st.error(f"An error occured: {e}")            