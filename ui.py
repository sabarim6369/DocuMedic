import streamlit as st
import os
from agent import PDFChatAgent

# ---- Page Config ----
st.set_page_config(
    page_title="📄 PDF Q&A with Groq",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- App Title ----
st.title("📄 Ask Your PDFs (Groq + LangChain)")
st.caption("Upload PDFs, ask questions, and get context-aware answers powered by Groq LLM.")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Upload PDFs")
    st.markdown("Upload one or more PDFs to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "📤 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Create agent
    agent = PDFChatAgent()
    file_paths = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp_" + uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_path)

        agent.load_pdfs(file_paths)
        st.success(f"✅ {len(file_paths)} PDF(s) loaded successfully!")

# ---- Main Content ----
st.markdown("## 🔍 Ask a Question")
query = st.text_input("Type your question here...")

if query:
    with st.spinner("🤖 Thinking..."):
        answer, docs = agent.ask(query)

    # Show Answer
    st.markdown("## 💡 Answer")
    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:12px;
            background-color:#f0f9ff;
            border:1px solid #d0eaff;
            color:#000000;
            font-size:16px;
            line-height:1.5;
        ">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )
