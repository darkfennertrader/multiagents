# %%
import asyncio
import functools
import operator
import uuid
from typing import Annotated, Sequence, TypedDict, List

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from tools.pdf import PDF_DIRECTORY
from tools.web import research
from web_research_prompts import RESEARCHER_SYSTEM_PROMPT, TAVILY_AGENT_SYSTEM_PROMPT

from IPython.display import Image
from langgraph.graph import END, StateGraph

from config import set_environment_variables

set_environment_variables("Web Search Graph")

TAVILY_TOOL = TavilySearchResults(max_results=1)
LLM = ChatOpenAI(model="gpt-4o")


TAVILY_AGENT_NAME = "tavily_agent"
RESEARCH_AGENT_NAME = "search_evaluator_agent"
SAVE_FILE_NODE_NAME = "save_file"


# helper to create agents
def create_agent(llm: ChatOpenAI, tools: List, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    # Agent Executor combines an agent and a list of tools in a single node
    agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
    return agent_executor


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state: AgentState, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


async def async_agent_node(state: AgentState, agent, name):
    result = await agent.ainvoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


tavily_agent = create_agent(LLM, [TAVILY_TOOL], TAVILY_AGENT_SYSTEM_PROMPT)
tavily_agent_node = functools.partial(
    agent_node, agent=tavily_agent, name=TAVILY_AGENT_NAME
)

research_agent = create_agent(LLM, [research], RESEARCHER_SYSTEM_PROMPT)
research_agent_node = functools.partial(
    async_agent_node, agent=research_agent, name=RESEARCH_AGENT_NAME
)


# the "save node" does not require an LLM
def save_file_node(state: AgentState):
    markdown_content = str(state["messages"][-1].content)
    filename = f"{PDF_DIRECTORY}/{uuid.uuid4()}.md"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    return {
        "messages": [
            HumanMessage(
                content=f"Output written successfully to {filename}",
                name=SAVE_FILE_NODE_NAME,
            )
        ]
    }


workflow = StateGraph(AgentState)
workflow.add_node(TAVILY_AGENT_NAME, tavily_agent_node)
workflow.add_node(RESEARCH_AGENT_NAME, research_agent_node)
workflow.add_node(SAVE_FILE_NODE_NAME, save_file_node)

workflow.add_edge(TAVILY_AGENT_NAME, RESEARCH_AGENT_NAME)
workflow.add_edge(RESEARCH_AGENT_NAME, SAVE_FILE_NODE_NAME)
workflow.add_edge(SAVE_FILE_NODE_NAME, END)

workflow.set_entry_point(TAVILY_AGENT_NAME)
research_graph = workflow.compile()

Image(research_graph.get_graph().draw_png())  # type: ignore

# %%


async def run_research_graph(_input):
    async for output in research_graph.astream(_input, stream_mode="updates"):
        for node_name, output_value in output.items():
            print(f"Output from node: {node_name}:")
            print(output_value)
            print("\n------------------\n")


test_input = {"messages": [HumanMessage(content="Python advanced courses")]}
asyncio.run(run_research_graph(test_input))
