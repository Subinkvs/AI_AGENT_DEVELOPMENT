from ddgs import DDGS
from langchain.tools import tool

from rag.rag_store import (
    EMBEDDINGS,
    INDEX_PATH,
    load_rag_retriever,
)
from langchain_community.vectorstores import FAISS


rag_retriever = load_rag_retriever()


@tool
def search_web(query: str) -> str:
    """
    Search the public web for current or external information.

    Official and primary sources should be preferred whenever possible.
    Returns titles, URLs, and snippets for source attribution.
    """
    try:
        results = DDGS().text(
            query,
            region="in-en",
            safesearch="moderate",
            max_results=8,
        )

        if not results:
            return "NO_WEB_RESULTS_FOUND"

        # Prefer authoritative domains.
        preferred_domains = [
            "python.org",
            "docs.python.org",
            "github.com",
            "microsoft.com",
            "google.com",
            "openai.com",
            "ollama.com",
            "arxiv.org",
            "wikipedia.org",
        ]

        def domain_priority(result):
            url = result.get("href", "").lower()

            for index, domain in enumerate(preferred_domains):
                if domain in url:
                    return index

            return len(preferred_domains)

        results = sorted(
            results,
            key=domain_priority,
        )

        output = [
            "WEB SEARCH RESULTS:",
            "Prefer authoritative/official sources when answering.",
            "",
        ]

        for result in results:
            title = result.get("title", "Untitled")
            url = result.get("href", "")
            body = result.get("body", "")

            output.append(
                f"TITLE: {title}\n"
                f"SOURCE_URL: {url}\n"
                f"CONTENT: {body}"
            )

        return "\n\n".join(output)

    except Exception as exc:
        return f"WEB_SEARCH_ERROR: {exc}"


@tool
def rag_search(query: str) -> str:
    """
    Search internal company documents.

    Returns the most relevant internal document chunks with their
    source filenames. Lower FAISS distance means a better match.
    """
    try:
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )

        results = vectorstore.similarity_search_with_score(
            query,
            k=5,
        )

        if not results:
            return "NO_RELEVANT_DOCUMENTS_FOUND"

        # Lower distance = better match.
        best_score = float(results[0][1])

        # Keep documents reasonably close to the best match.
        max_allowed_score = best_score + 0.25

        selected = [
            (doc, float(score))
            for doc, score in results
            if float(score) <= max_allowed_score
        ]

        output = []
        seen_sources = set()

        for doc, score in selected:
            source = doc.metadata.get("source", "unknown")

            if source in seen_sources:
                continue

            seen_sources.add(source)

            output.append(
                f"SOURCE: {source}\n"
                f"SCORE: {score:.4f}\n"
                f"CONTENT: {doc.page_content}"
            )

        return "\n\n".join(output)

    except Exception as exc:
        return f"RAG_SEARCH_ERROR: {exc}"