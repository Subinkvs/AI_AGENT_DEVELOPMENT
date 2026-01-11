from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool

duck = DuckDuckGoSearchRun()

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def search_web(query: str) -> str:
    """
    Search the web and Wikipedia for factual information.
    Returns content along with exact sources.
    """
    wiki_result = wiki_tool.run(query)
    duck_result = duck.run(query)

    return f"""
SOURCE: Wikipedia
{wiki_result}

SOURCE: DuckDuckGo
{duck_result}
"""




