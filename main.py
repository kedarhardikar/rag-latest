import streamlit as st
import tempfile
import os
from graph_builder import build_graph
from nodes.upload_node import calculate_file_hash

st.set_page_config(page_title="🧠 Multi-Modal LangGraph Agent")
st.title("📄🖼️ LangGraph: RAG + VQA Agent")

SUPPORTED_TYPES = ["pdf", "txt", "docx", "md", "png", "jpg", "jpeg", "csv", "xlsx"]

# Upload file (document or image)
uploaded_file = st.file_uploader("Upload a document or image", type=SUPPORTED_TYPES)

file_path = None
# image_path = None

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
        if suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_path = file_path

    st.success(f"✅ Uploaded: {uploaded_file.name}")
    file_hash = calculate_file_hash(file_path)
    st.info(f"File hash: `{file_hash}`")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Ask question
    user_input = st.text_input("💬 Ask a question about the uploaded file:")

    if user_input:
        with st.spinner("🔍 Reasoning..."):
            graph = build_graph()
            result = graph.invoke({
            "input": user_input,
            "file_path": file_path,
            "chat_history": st.session_state.chat_history
        })


            st.write("### ✅ Answer:")
            st.markdown(result["answer"])
