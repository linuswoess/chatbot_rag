import streamlit as st
import os

# Import LangChain and related modules
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set page config
st.set_page_config(page_title="Question Answering with Groq", layout="wide")

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Sidebar for API key input
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Groq API Key:", type="password", key="groq_api_key_input")
    
    if api_key:
        st.session_state.api_key = api_key
        os.environ['GROQ_API_KEY'] = api_key
        st.success("API Key configured!")
    elif not st.session_state.api_key:
        st.warning("Please enter your Groq API Key to proceed")

# Check if API key is available before proceeding
if not st.session_state.api_key:
    st.error("🔑 Please enter your Groq API Key in the sidebar to use this app")
    st.stop()

# Load CSV document
loader = CSVLoader(r'fake_startup_founders_europe.csv', encoding="latin-1")

doc = loader.load()

# Create embeddings for all the documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in vectordatabase
vectorstore = FAISS.from_documents(doc, embeddings)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
)

# Setting up the retrieval function using modern LCEL approach
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Question Answering with Groq Chat")

# User input for the query
question = st.text_input("Enter your question:")

if st.button("Submit") and question:
    with st.spinner("Processing..."):
        result = chain.invoke(question)
    st.write("**Result:**")
    st.write(result)
