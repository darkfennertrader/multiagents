# %%
import os
from time import time, sleep
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from langgraph.graph import MessageGraph, StateGraph, START, END


from config import set_environment_variables

set_environment_variables("langgrapg_async")


class State(TypedDict):
    which: str
    aggregate: Annotated[List, operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str) -> None:
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


builder = StateGraph(State)

builder.add_edge(START, "a")
builder.add_node("a", ReturnNodeValue("node A"))
builder.add_node("b", ReturnNodeValue("node B"))
builder.add_node("c", ReturnNodeValue("node C"))
builder.add_node("d", ReturnNodeValue("node D"))
builder.add_node("e", ReturnNodeValue("node E"))

builder.add_edge("b", "e")
builder.add_edge("c", "e")
builder.add_edge("d", "e")
builder.add_edge("e", END)

# this is necessary if you want to avoid unnecessary links in the graph
intermediates = ["b", "c", "d"]
builder.add_conditional_edges("a", route_bc_or_cd, intermediates)

graph = builder.compile()


file_path = "langgraph_parallel_3.png"
if not os.path.exists(file_path):
    display(Image(graph.get_graph().draw_mermaid_png(output_file_path=file_path)))

# %%

if __name__ == "__main__":
    start = time()
    graph.invoke({"aggregate": [], "which": "cd"}, {"configurable": {"thread_id": "1"}})
    print(f"\nParallel execution took {time() - start:.2f} sec.")
