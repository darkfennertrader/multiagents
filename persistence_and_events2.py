# %%
import operator
from typing import TypedDict, Annotated, Dict, Any
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph, StateGraph, START, END

from config import set_environment_variables

set_environment_variables("persistence")

memory = MemorySaver()
# saving to local disk
memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

tool = TavilySearchResults(max_results=2)
# print(tool.name, type(tool))


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State):
    print("\n--- Step 1 ---")


def human_feedback(state: State):
    print("\n--- human_feedback ---")


def step_3(state: State):
    print("\n--- Step 3 ---")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

display(Image(graph.get_graph().draw_mermaid_png(output_file_path="persistence.png")))  # type: ignore

# %%

if __name__ == "__main__":

    thread = {"configurable": {"thread_id": "1"}}
    initial_input = {"input": "Hello World!"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):  # type: ignore
        print(event)

    print(graph.get_state(thread).next)  # type: ignore

    user_input = input("Tell me how you want to update the state: ")

    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")  # type: ignore

    print(graph.get_state(thread).next)  # type: ignore

    # print StateSnapshot
    print(graph.get_state(thread))  # type: ignore

    # let's continue to stream our graph
    for event in graph.stream(None, thread, stream_mode="values"):  # type: ignore
        print(event)
