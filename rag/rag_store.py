import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEX_PATH = "rag/faiss_index"


def build_rag_index():
    documents = []
    for file in os.listdir("rag/docs"):
        if file.endswith(".txt"):
            loader = TextLoader(f"rag/docs/{file}")
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    vectorstore.save_local(INDEX_PATH)


def load_rag_retriever():
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("FAISS index not found. Building index...")
        build_rag_index()

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

