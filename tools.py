from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.tools import tool
from rag.rag_store import load_rag_retriever

duck = DuckDuckGoSearchRun()
rag_retriever = load_rag_retriever()



@tool
def search_web(query: str) -> str:
    """
    Search the web and Wikipedia for factual information.
    Returns content along with exact sources.
    """
    duck_result = duck.run(query)

    return f"""

SOURCE: DuckDuckGo
{duck_result}
"""

@tool
def rag_search(query: str) -> dict:
    """
    Search internal company documents using vector similarity.
    Returns context and exact document sources.
    """
    docs = rag_retriever.invoke(query)

    if not docs:
        return {
            "context": "",
            "sources": []
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in docs
    })

    return {
        "context": context,
        "sources": sources
    }


