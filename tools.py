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
def rag_search(query: str) -> str:
    """
    Search internal company documents using vector similarity.
    """
    docs = rag_retriever.invoke(query)

    if not docs:
        return "No relevant internal documents found."

    context = "\n\n".join(doc.page_content for doc in docs)

    return f"""
SOURCE: Internal Knowledge Base
{context}
"""


