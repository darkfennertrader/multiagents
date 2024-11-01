import asyncio
from time import sleep
from typing import Annotated
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableGenerator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def my_generator(state: State):
    messages = [
        "Four",
        "score",
        "and",
        "seven",
        "years",
        "ago",
        "our",
        "fathers",
        "...",
    ]
    for message in messages:
        yield message
        sleep(0.2)


async def my_node(state: State):
    messages = []
    # Tagging a node makes it easy to filter out which events to include in your stream
    # It's completely optional, but useful if you have many functions with similar names
    gen = RunnableGenerator(my_generator).with_config(tags=["should_stream"])  # type: ignore
    async for message in gen.astream(state):
        messages.append(message)
    return {"messages": [AIMessage(content=" ".join(messages))]}


workflow = StateGraph(State)
workflow.add_node("model", my_node)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)
app = workflow.compile()


inputs = [HumanMessage(content="What are you thinking about?")]


async def main():
    async for event in app.astream_events({"messages": inputs}, version="v1"):
        kind = event["event"]
        print("\n", "-" * 80, "\n", flush=True)
        print(event, flush=True)
        print("-" * 80, "\n", flush=True)
        tags = event.get("tags", [])
        if kind == "on_chain_stream" and "should_stream" in tags:
            data = event["data"]
            if data:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(data, end="|")


asyncio.run(main())
