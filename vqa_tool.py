# vqa_tool.py (Updated to use PaddleOCR's predict() method)

from PIL import Image
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
# --- PaddleOCR Imports and Initialization ---
from paddleocr import PaddleOCR
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)  # Or use "llama3-70b-8192" etc.


# --- VQA Tool Node (using OCR + LLM Placeholder) ---

def vqa_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    
    """
    Langgraph node for performing Visual Question Answering using PaddleOCR
    for text extraction and an LLM placeholder for reasoning.
    """
    
    print("Initializing PaddleOCR with default English models on CPU...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,           # Use angle classification to handle rotated text lines
        lang="en"                # Language model
    )
    print("PaddleOCR engine initialized.")

    question = state.get("input", "")
    file_path = state.get("file_path", "")

    if not question or not file_path:
        print("⚠️ Tool: Missing question or file path in state.")
        return {**state, "answer": "⚠️ Requires both a question and an image file path."}

    if not os.path.exists(file_path):
        return {**state, "answer": f"Error: Image file not found at {file_path}"}

    try:
        print(f"Starting OCR for image: {file_path}")
        # The .predict() method returns a comprehensive result dictionary. This automatically handles detection, cropping, and recognition.
        ocr_result = ocr_engine.predict(file_path)
        # print(f"OCR result: {ocr_result}")  #Contains [rec_texts] which is the desired output
        extracted_texts = []
        
        extracted_texts = ocr_result[0]['rec_texts']
        print(f"OCR completed. Total extracted text lines: {len(extracted_texts)}")

        if len(extracted_texts) == 0:
            return {**state, "answer": "OCR found no text regions in the image."}

        # Join all extracted text lines into a single string for the LLM
        full_extracted_text = " ".join(extracted_texts)

        if not full_extracted_text.strip():
            return {**state, "answer": "OCR found text regions but extracted no readable text."}

        print(f"OCR extracted text (first 100 chars): '{full_extracted_text[:100]}...'")

        # 2. Formulate a prompt for the text-based LLM (RAG)
        rag_prompt = (
                    f"Based on the following text extracted from a diagram, directly and concisely answer the question. "
                    f"Your response MUST be ONLY the answer. Do NOT include any explanations, reasoning, conversational phrases, "
                    f"or introductory/concluding remarks. "
                    f"If the information required to answer the question is not explicitly present in the 'Diagram Text', "
                    f"you MUST respond with: 'The information is not explicitly present in the provided text.'\n\n"
                    f"Diagram Text:\n{full_extracted_text}\n\n"
                    f"Question: {question}\n\nAnswer:"
                )

        # 3. Send to LLM (Placeholder)

        response = llm.invoke(rag_prompt)  # Using .invoke() for runnable
        response_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        return {**state, "answer": response_content}

    except Exception as e:
        print(f"❌ OCR Tool Error: {e}")
        import traceback
        traceback.print_exc()
        return {**state, "answer": f"Error processing with OCR: {e}"}