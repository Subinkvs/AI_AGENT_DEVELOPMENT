import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

INDEX_PATH = "rag/faiss_index"


def build_rag_index():
    documents = []
    for file in os.listdir("rag/docs"):
        if file.endswith(".txt"):
            loader = TextLoader(f"rag/docs/{file}")
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    vectorstore.save_local(INDEX_PATH)


def load_rag_retriever():
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})
