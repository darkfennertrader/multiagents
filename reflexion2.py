# %%
import os
from datetime import datetime
from time import time
from typing import TypedDict, List, Annotated
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from config import set_environment_variables

set_environment_variables("Reflexion_2")

memory = SqliteSaver.from_conn_string(":memory:")
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# planning prompt
PLAN_PROMPT = """
You are an expert writer tasked with writing a high level ourline of an essay.
Wrtite such an outline for the user provided topic.
Give an outline of the essay along with any relevant notes or instructions for the sections.
"""

# research prompt
RESEARCH_PLAN_PROMPT = f"""
You are a researcher charged with providing information that can be used when writing the following essay, Generate a list of search queries that will gather any relevant information. Only generate 3 queries max. Always choose the most up-to-dated queries.
\nCURRENT DATE
{datetime.now().isoformat()}
"""

# writer prompt
WRITER_PROMPT = """
You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possibile for the user's reequest and the initial outline.
If the user provides critique, respond with a revised version of your previous attempts.
Utilize all the information below as needed:

-----

{content}
"""

# reflection prompt
REFLECTION_PROMPT = """
You are a techer grading an essay submission.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth, style, etc.
"""

# research critique prompt
RESEARCH_CRITIQUE_PROMPT = f"""
You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below).
Generate a list of search queries that will gather relevant information. Only generate 3 queries max. Always choose the most up-to-dated queries.
\nCURRENT DATE
{datetime.now().isoformat()}
"""


class Queries(BaseModel):
    queries: List[str]


def plan_node(state: AgentState):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ]
    queries = model.with_structured_output(Queries).invoke(messages)

    content = state["content"] or []
    for q in queries.queries:  # type: ignore
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:  # type: ignore
            content.append(r["content"])
    return {"content": content}


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


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["critique"]),
    ]
    queries = model.with_structured_output(Queries).invoke(messages)

    content = state["content"] or []
    for q in queries.queries:  # type: ignore
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:  # type: ignore
            content.append(r["content"])
    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
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

display(Image(app.get_graph().draw_mermaid_png(output_file_path="reflexion_2.png")))  # type: ignore

# %%

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    task = "How to structure a report for feedback after a course on soft skills."
    input = {"task": task, "revision_number": 1, "max_revisions": 3}

    start = time()
    for event in app.stream(input, stream_mode="updates", config=config):  # type: ignore
        print()
        print(event)  # type: ignore
        print("-" * 80, "\n")

    print(f"\nIt took {(time() -start):.2f} sec. to evaluate your request")
