# %%
import os
from time import time
import asyncio
import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel
from IPython.display import Image, display
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import MessageGraph, StateGraph, END

from config import set_environment_variables

set_environment_variables("essay_writer")

memory = SqliteSaver.from_conn_string(":memory:")

# tool = TavilySearchResults(max_results=3)
tavily = TavilyClient(api_key="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")
model = ChatOpenAI(model="gpt-4o", temperature=0)


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes or instructions for the sections."""


def plan_node(state: AgentState):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])]
    response = model.invoke(messages)
    return {"plan": response.content}


RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can be used when writing the following essay. Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:  # type: ignore
        response = tavily.search(query=q, max_results=2)
        if response:
            for r in response["results"]:
                content.append(r["content"])

    return {"content": content}


WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""


def generation_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        HumanMessage(content=user_message),
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["critique"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:  # type: ignore
        response = tavily.search(query=q, max_results=2)
        if response:
            for r in response["results"]:
                content.append(r["content"])

    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate", should_continue, {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
app = builder.compile(checkpointer=memory)

# display(Image(app.get_graph().draw_png()))  # type: ignore

# %%

if __name__ == "__main__":

    inputs = {
        "task": "Write about the current European elections.",
        "max_revisions": 2,
        "revision_number": 1,
    }

    thread = {"configurable": {"thread_id": "1"}}

    # with stream it displays all the passages  between agents
    start = time()
    for s in app.stream(inputs, thread):  # type: ignore
        print()
        print(s)
        print("-" * 50)
    print(f"\nOverall time: {(time() - start):.2f} sec.")
