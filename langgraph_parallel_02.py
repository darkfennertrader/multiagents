# %%
import os
from time import time, sleep
import operator
from typing import TypedDict, Annotated, List, Dict, Any
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
    aggregate: Annotated[List, operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str) -> None:
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


builder = StateGraph(State)

builder.add_node("a", ReturnNodeValue("node A"))
builder.add_node("b1", ReturnNodeValue("node B1"))
builder.add_node("b2", ReturnNodeValue("node B2"))
builder.add_node("c", ReturnNodeValue("node C"))
builder.add_node("d", ReturnNodeValue("node D"))

builder.add_edge(START, "a")
builder.add_edge("a", "b1")
builder.add_edge("a", "c")
builder.add_edge("b1", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()


file_path = "langgraph_parallel_2.png"
if not os.path.exists(file_path):
    display(Image(graph.get_graph().draw_mermaid_png(output_file_path=file_path)))

# %%

if __name__ == "__main__":
    start = time()
    graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "1"}})
    print(f"\nParallel execution took {time() - start:.2f} sec.")
