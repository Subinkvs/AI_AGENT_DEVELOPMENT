from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_web, wiki_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

tools=[search_web, wiki_tool]

agent = create_agent(
    model=llm,
    system_prompt=f"""
You are a research assistant.

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
    tools= tools
)

query= input("What can i help you research ?\n")

raw_response = agent.invoke({
    "messages": [{"role": "user", "content": query}]
})

ai_message = raw_response["messages"][-1]


if isinstance(ai_message.content, list):
    raw_text = ai_message.content[0]["text"]
else:
    raw_text = ai_message.content

structured_response = parser.parse(raw_text)

print(structured_response)




