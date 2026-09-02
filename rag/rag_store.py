from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = str(BASE_DIR / "faiss_index")


EMBEDDINGS = OllamaEmbeddings(
    model="nomic-embed-text"
)



def build_rag_index():
    documents = []

    for file_path in DOCS_DIR.glob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    if not documents:
        raise RuntimeError(
            f"No .txt documents found in {DOCS_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        chunks,
        EMBEDDINGS,
    )

    vectorstore.save_local(INDEX_PATH)

    print(f"RAG index built with {len(chunks)} chunks")


def load_rag_retriever():
    index_file = Path(INDEX_PATH) / "index.faiss"

    if not index_file.exists():
        print("FAISS index not found. Building index...")
        build_rag_index()

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )