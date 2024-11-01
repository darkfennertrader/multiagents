# %%
import asyncio
import operator
from uuid import uuid4
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
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph, StateGraph, END

from config import set_environment_variables

set_environment_variables("react_from_basics")

memory = SqliteSaver.from_conn_string(":memory:")


"""
In previous examples, we've annotated the messages' state key with the default 'operator.add' or '+' reducer, which always append new messages to the end of existing message array.
Now, to support replacing exisitng messages, we annotate the 'messages' key witha  customer reducer function, which replaces messages with the same 'id', and appends them otherwise
"""


def reduce_messages(
    left: list[AnyMessage], right: list[AnyMessage]
) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged


tool = TavilySearchResults(max_results=2)
# print(tool.name, type(tool))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class Agent:
    def __init__(
        self, model, tools, checkpointer, system="", human_in_the_loop: bool = False
    ) -> None:
        self.system = system
        self.human_in_the_loop = human_in_the_loop
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exist_action, {True: "action", False: END}  # type: ignore
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.app = (
            graph.compile(checkpointer=checkpointer, interrupt_before=["action"])
            if self.human_in_the_loop
            else graph.compile(checkpointer=checkpointer)
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            message = self.model.invoke(messages)
            return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))  # type: ignore
            )
        print("Back to the model")
        return {"messages": results}

    def exist_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0  # type: ignore


if __name__ == "__main__":
    prompt = """You are a smart research assistant. Use the search engine to look up information. You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(model="gpt-4o")
    ai_bot = Agent(
        model, [tool], checkpointer=memory, system=prompt, human_in_the_loop=True
    )

    query = "What's the weather in LA"
    messages = [HumanMessage(content=query)]
    # different threads inside the checkpointer for multiple conversation
    thread = {"configurable": {"thread_id": "1"}}

    for event in ai_bot.app.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)

    while ai_bot.app.get_state(thread).next:
        print("\n", ai_bot.app.get_state(thread), "\n")
        _input = input("proceed?")
        if _input != "y":
            print("aborting")
            break
        for event in ai_bot.app.stream(None, thread):
            for v in event.values():
                print(v)

    # print("\nCurrent state of the graph:")
    # print(ai_bot.app.get_state(thread))

    # print("\nNext node: ", ai_bot.app.get_state(thread).next)
    # is_ok = bool(input("is it ok with you (y/n)?"))
    # if is_ok:
    #     for event in ai_bot.app.stream(None, thread):
    #         for v in event.values():
    #             print(v)
