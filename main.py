from dotenv import load_dotenv
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 
from tools import search_web, rag_search

load_dotenv()

app = FastAPI(title="Agentic RAG Backend API")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class AskAPIResponse(BaseModel):
    answer: str
    source: List[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

tools=[search_web, rag_search]

agent = create_agent(
    model=llm,
    system_prompt=f"""
You are a research assistant.

Use:
- rag_search for internal/company/product/technical documentation
- search_web for external or real-time information


You MUST return your final answer as VALID JSON.
The JSON MUST strictly follow this schema:

{parser.get_format_instructions()}

Rules (MANDATORY):
- Output ONLY JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include extra text
- Populate "tools_used" with the names of tools you used
- If no tools are used, return an empty list
""",
    tools= tools,
    checkpointer=InMemorySaver(), 
)



@app.post("/ask", response_model=AskAPIResponse)
def ask(payload: AskRequest):
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": payload.query}]},
        {"configurable": {"thread_id": payload.session_id}},
    )

    ai_message = raw_response["messages"][-1]

    if isinstance(ai_message.content, list):
        raw_text = ai_message.content[0]["text"]
    else:
        raw_text = ai_message.content

    structured_response: ResearchResponse = parser.parse(raw_text)

    return {
        "answer": structured_response.summary,
        "source": structured_response.sources
    }