#rag_tool.py
import re
from typing import Dict
from langchain.schema import SystemMessage
from langchain.schema.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)


def rag_tool_node(state: Dict) -> Dict:
    import logging
    logging.basicConfig(level=logging.INFO)

    query = state["input"]
    vectordb = state.get("vectordb")
    chat_history = state.get("chat_history", [])

    if vectordb is None:
        print("⚠️ RAG Tool: Document database not found in state.")
        return {
            **state,
            "answer": "⚠️ Document database not available for RAG. Please ensure a document was uploaded successfully."
        }

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    top_docs = retriever.invoke(query)

    if not top_docs:
        print("⚠️ RAG Tool: No relevant information found.")
        return {
            **state,
            "answer": "⚠️ No relevant information found in the documents.",
            "chat_history": chat_history  # Keep history intact
        }

    context = "\n\n".join([doc.page_content for doc in top_docs])

    # Build message history for LLM input
    messages = [
        SystemMessage(content=(
            "You are a strict and factual assistant. Only answer questions if the information is explicitly present "
            "in the provided documents. Do not use outside knowledge, assumptions, or general reasoning. "
            "If the answer is not directly found in the documents, respond with: 'Irrelevant docs uploaded.' "
            "Do not try to guess or provide unrelated information. Stay within the source material only."
        )),
    ]

    messages.extend(chat_history)  # Add past turns

    # Add current question
    messages.append(HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"))

    # Run LLM
    response = llm.invoke(messages)
    raw_response = response.content
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    # Log the final cleaned output
    logging.info(f"LLM Response from RAG: {cleaned}")

    # Update memory with current turn
    chat_history.append(HumanMessage(content=query))
    chat_history.append(response)

    return {
        **state,
        "answer": cleaned,
        "chat_history": chat_history
    }


    import logging
    # basicConfig only sets up logging if no handlers are configured.
    # If you've configured it elsewhere (e.g., in main.py), this might not re-configure it.
    logging.basicConfig(level=logging.INFO)
    logging.info(f"LLM Response from RAG: {response}") # Added prefix for clarity in logs

    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    return {**state, "answer": cleaned}