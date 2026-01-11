from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import AnthropicLLM, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_agent(
    model=llm,
    system_prompt="""
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output in the specified format and provide no other text.
""",
    tools=[]
)


# agent_executer = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent.invoke({
    "messages": [{"role": "user", "content": "What is an AI Agent ?"}]
})
print(raw_response["messages"][-1])




