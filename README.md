# 🧠 LangGraph MultiTool Agent with RAG & Tool-Use

This is an advanced conversational chatbot powered by **LangGraph**, **OpenAI GPT-4o-mini** (via OpenRouter), **TavilySearch** for real-time tool access, and **Retrieval-Augmented Generation (RAG)** from a PDF document.

It dynamically chooses between using tools or retrieving answers from a custom document to enhance responses, making it a powerful AI assistant.

---

## 🚀 Features

- ✅ **ChatGPT with Tools**: Uses `TavilySearch` to answer real-time or general questions.
- ✅ **PDF Document Retrieval (RAG)**: Loads and indexes `nike.pdf`, retrieves the most relevant chunks to assist the LLM.
- ✅ **LangGraph Flow Control**: Manages flow between tool use and RAG-based retrieval using conditional edges.
- ✅ **HuggingFace Embeddings**: Embeds document text using `sentence-transformers/all-mpnet-base-v2`.
- ✅ **State-Based Chat Loop**: Fully interactive terminal-based chatbot with contextual memory.
- ✅ **Conditional Reasoning**: Decides whether to use tools or retrieve document context based on user query.

---

## 🧱 Tech Stack


| **LangGraph** | Framework to model stateful multi-step interactions |
| **OpenRouter (GPT-4o-mini)** | LLM provider for chat completion |
| **TavilySearch Tool** | Real-time web search tool |
| **PyPDFLoader** | Loads PDF content |
| **RecursiveCharacterTextSplitter** | Splits long text into manageable chunks |
| **HuggingFace Embeddings** | For semantic similarity search |
| **InMemoryVectorStore** | Lightweight vector DB for document retrieval |



