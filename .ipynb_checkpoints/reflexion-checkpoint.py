from time import time
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import revisor, first_responder
from tool_executor import execute_tools

from config import (
    OPENAI_API_KEY,
    TAVILY_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
)

MAX_ITERATIONS = 2
builder = MessageGraph()

builder.add_node(key="draft", action=first_responder)
builder.add_node(key="execute_tools", action=execute_tools)
builder.add_node(key="revise", action=revisor)

builder.add_edge(start_key="draft", end_key="execute_tools")
builder.add_edge(start_key="execute_tools", end_key="revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END

    return "execute_tools"


builder.add_conditional_edges(start_key="revise", condition=event_loop)
builder.set_entry_point("draft")

graph = builder.compile()


if __name__ == "__main__":

    start = time()
    task = "Write a report on the upcoming P500 GMT game: 'The Pure Land: Onin War in Morumachi Japan'"
    print("Please wait while evaluating your request...")
    response = graph.invoke(task)
    print()
    print(response[-1].tool_calls[0]["args"]["answer"])
    print(f"\nIt took {(time() -start):.2f} sec. to evaluate your request")
