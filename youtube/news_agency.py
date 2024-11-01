# %%

import functools
import operator
from typing import Annotated, Sequence, TypedDict, List
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from IPython.display import Image, display
from config import set_environment_variables

set_environment_variables("MultiAgent_NewsAgency")


SUPERVISOR = "supervisor"
NEWS_CORRESPONDENT = "news_correspondent"
NEWS_EDITOR = "news_editor"
ADS_WRITER = "ads_writer"

DDG = DuckDuckGoSearchAPIWrapper(max_results=5)
LLM = ChatOpenAI(model="gpt-4-turbo-2024-04-09")


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet"""
    response = requests.get(url=url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


@tool("internet_search_tool", return_direct=False)
def internet_search_tool(query: str) -> str:
    """Search provided query on the internet using DuckDuckGo"""
    results = DDG.run(query)
    return results if results else "No results found."


tools = [internet_search_tool, process_search_tool]


# helper to create agents
def create_agent(llm: BaseChatModel, tools: List, system_prompt: str) -> AgentExecutor:
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


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = [NEWS_CORRESPONDENT, NEWS_EDITOR, ADS_WRITER]
options = ["FINISH"] + members

system_prompt = system_prompt = (
    "As a supervisor, your role is to oversee the insight between these:"
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress."
    " Once all tasks are completed, indicate 'FINISH'."
)

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | LLM.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

news_correspondent_agent = create_agent(
    LLM,
    tools,
    """Your primary role is to function as an intelligent news research assistant, adept at scouring the internet for the latest and most relevant trending stories across various sectors like politics, technology, health, culture, and global events. You possess the capability to access a wide range of online news sources, blogs, and social media platforms to gather real-time information.""",
)

news_correspondent_node = functools.partial(
    agent_node, agent=news_correspondent_agent, name=NEWS_CORRESPONDENT
)

news_editor_agent = create_agent(
    LLM,
    tools,
    """You are a news editor. Do step by step approach.
    Based on the provided content first identify the list of topics, then search internet for each topic one by one and finally find insights for each topic one by one that can aid you in writting a useful news edition for AI-nes corp. Include the insights and sources in the final response.""",
)
# changed from news_editor_node => news_editor
news_editor_node = functools.partial(
    agent_node, agent=news_editor_agent, name=NEWS_EDITOR
)

ads_writer_agent = create_agent(
    LLM,
    tools,
    """You are an ads writter for AI-news corp. Given the publication generated by the
    news editor, your work if to write ads that relate to that content. Use the internet 
    to search for content to write ads based off on. Here is a description of your task:
    
    To craft compelling and relevant advertisements for 'AI News' publication, complementing the content written by the news editor.
    Contextual Ad Placement: Analyze the final report content from the news editor in-depth to identify key themes, topics, 
    and reader interests. Place ads that are contextually relevant to these findings, thereby increasing potential customer engagement.
    Advanced Image Sourcing and Curation: Employ sophisticated web search algorithms to source high-quality, relevant images for each ad. 
    Ensure these images complement the ad content and are aligned with the publication's aesthetic standards.
    Ad-Content Synchronization: Seamlessly integrate advertisements with the report, ensuring they enhance rather than disrupt the reader's 
    experience. Ads should feel like a natural extension of the report, offering value to the reader.
    Reference and Attribution Management: For each image sourced, automatically generate and include appropriate references and attributions, 
    ensuring compliance with copyright laws and ethical standards.
    """,
)
ads_writer_node = functools.partial(agent_node, agent=ads_writer_agent, name=ADS_WRITER)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# Create workflow or graph
workflow = StateGraph(AgentState)
# adding nodes
workflow.add_node(SUPERVISOR, supervisor_chain)
workflow.add_node(NEWS_CORRESPONDENT, news_correspondent_node)
workflow.add_node(NEWS_EDITOR, news_editor_node)
workflow.add_node(ADS_WRITER, ads_writer_node)

for member in members:
    workflow.add_edge(member, SUPERVISOR)

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

# if task is FINISHED, supervisor won't send task to agent, else,
# the supervisor will keep on sending task to agent untill done, this is
# what the conditional edge does.
workflow.add_conditional_edges(SUPERVISOR, lambda x: x["next"], conditional_map)
workflow.set_entry_point(SUPERVISOR)
# print(workflow.branches)
# print(workflow.edges)
# print(workflow.nodes)
# print(workflow.channels)
graph = workflow.compile()


Image(graph.get_graph().draw_png())  # type: ignore

# %%

if __name__ == "__main__":
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="""Write me a report on Italian maritime shipping sector. After the research on the sector, pass the findings to the news editor to generate the final publication. Once done, pass it to the ads writter to write the ads on the subject."""
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    ):
        if not "__end__" in s:
            print(s, end="\n\n-----------------\n\n")
