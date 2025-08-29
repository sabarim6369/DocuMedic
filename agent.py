import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class PDFChatAgent:
    def __init__(self):
        # Initialize embeddings and LLM
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        self.db = None

    def load_pdfs(self, file_paths):
        """Load and process multiple PDFs into ChromaDB"""
        all_docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        # Create new Chroma DB (in-memory for now)
        self.db = Chroma(embedding_function=self.embeddings, persist_directory=None)
        self.db.add_documents(splits)

    def ask(self, query, top_k=3):
        """Ask a question and get an LLM response"""
        if not self.db:
            return "⚠️ Please upload PDFs first."

        retriever = self.db.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)

        # Build context
        context = "\n\n".join([d.page_content for d in relevant_docs[:top_k]])
        prompt = f"""
You are a medical assistant AI.
Answer the user's question ONLY based on the provided context from medical documents.  
If the answer is not in the context, say "⚠️ I cannot find this information in the uploaded medical documents."  

User Question: {query}

Medical Context:
{context}
"""

        # Query LLM
        response = self.llm.invoke(query + "\n\nContext:\n" + context)
        return response.content, relevant_docs
