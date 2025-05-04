## 🚀 Coding Assistant: local RAG powered Development companion

Tired of sifting through documentation or online forums for coding help? Meet the **Coding Assistant**, a powerful tool designed to bring relevant knowledge directly to your fingertips, running entirely on local machine

---

### ✨ Goal

The core mission? To significantly **enhance the capabilities of a Large Language Models (LLM)** by building a robust **Retrieval Augmented Generation (RAG)** system.

---

### 🛠️ How It Works (Functionality)

This assistant is built around several key components working in harmony:

* 💾 **Indexing & Knowledge Base:**
    * Ingests knowledge from specific, authoritative sources (currently **Python PEP8**, **Google Python Style Guide**,  and **Google ML Rules**).
    * Stores this knowledge persistently in a **Vector Store** for efficient retrieval.

* 🧠 **Intelligent Runtime:**
    * Ready to tackle toughest challanges on **coding problems**, **Machine Learning tasks**, and **Algorithm implementations**.

* 🚦 **Smart Routing:**
    * When I ask a question, the system application decides:
        * Should it query the **internal knowledge base** (Vector Store)?
        * Does it need to perform a **web search** (using **Tavily**)?

* 💡 **Contextual Generation:**
    * Leverages a **locally running LLM (Gemma 3 via Ollama)**.
    * Generates precise solutions and answers based on the **retrieved context**, whether it came from curated knowledge base or a real-time web search

---

### 🏗️ The Tech Stack

Built with open source:

* **Framework:** Langchain - Orchestrating the RAG pipeline
* **LLM Server:** Ollama - running Google's large language model Gemma 3
* **Embedding Model:** Nomic Embeddings - generating vectors for local inference
* **Vector Store:** Open Source Vector Database (*📝 To be Implemented*) - for persistent knowledge storage
* **Web Search API:** Tavily API - intelligent web queries
* **User Interface:** Streamlit - providing an intuitive UI/UX

---

I aim to provide simple, free from hallucinations, and secure coding assistant by keeping everything running locally