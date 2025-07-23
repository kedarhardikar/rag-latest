# 🧠 LangGraph Multi-Modal RAG + VQA Agent

This is a **Streamlit-based interactive assistant** built using **LangGraph**, capable of answering user questions from uploaded **documents, images, and spreadsheets**. It combines the power of **ChromaDB**, **Groq LLMs**, **PaddleOCR**, and **LangChain** into a modular, node-based LangGraph workflow.

---

## ✅ Features

- 📄 **Document QA (RAG)** – Supports `.pdf`, `.docx`, `.txt`, `.md` using **ChromaDB** and **Groq LLM**
- 🖼️ **Image QA (VQA)** – Supports `.png`, `.jpg`, `.jpeg` using **PaddleOCR + Groq LLM**
- 📊 **Tabular QA** – Supports `.csv`, `.xlsx` via row-wise text conversion
- 🧠 **Memory** – Maintains session-level context using `chat_history`
- 💾 **Persistent Storage** – Avoids redundant embeddings using local **ChromaDB**

---

## 🧱 LangGraph Workflow (Node-based Breakdown)

| Node               | Description                                         |
|--------------------|-----------------------------------------------------|
| `ask_node.py`       | Receives and logs user input                       |
| `upload_node.py`    | Detects file type and manages ChromaDB collections |
| `route_mode_node.py`| Routes to `rag_tool` or `vqa_tool` based on input  |
| `rag_tool.py`       | Retrieves top-k document chunks + runs Groq LLM    |
| `vqa_tool.py`       | Runs OCR + Groq LLM for image-based QA             |
| `chat_history`      | Stored in `st.session_state`, passed between nodes |

---

## 🧾 Supported File Types

| Type   | Extensions                    |
|--------|-------------------------------|
| Text   | `.pdf`, `.txt`, `.docx`, `.md`|
| Image  | `.png`, `.jpg`, `.jpeg`       |
| Table  | `.csv`, `.xlsx`               |

---

## 🗂️ Folder Structure

```plaintext
.
├── main.py                    # Streamlit UI
├── graph_builder.py           # LangGraph DAG setup
├── rag_tool.py                # Handles RAG Q&A
├── vqa_tool.py                # Handles OCR + image-based Q&A
├── nodes/
│   ├── ask_node.py            # User input logging
│   ├── upload_node.py         # ChromaDB upload & file processing
│   ├── route_mode_node.py     # File-type-based routing
├── chroma_db_files/           # Persistent ChromaDB vector DB
├── .env                       # API keys (GROQ_API_KEY)
├── .gitignore                 # Ignored files (envs, cache, chroma, etc.)
├── req.txt                    # Python dependencies
└── README.md                  # Documentation
```
