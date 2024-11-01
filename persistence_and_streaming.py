# %%
import asyncio
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
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import MessageGraph, StateGraph, END

from config import set_environment_variables

set_environment_variables("react_from_basics")

memory = AsyncSqliteSaver.from_conn_string(":memory:")

tool = TavilySearchResults(max_results=2)
# print(tool.name, type(tool))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system="") -> None:
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exist_action, {True: "action", False: END}  # type: ignore
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.app = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    async def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            message = await self.model.ainvoke(messages)
            return {"messages": [message]}

    async def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = await self.tools[t["name"]].ainvoke(t["args"])
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
    ai_bot = Agent(model, [tool], checkpointer=memory, system=prompt)

    query = "Who is the weather in sf"
    messages = [HumanMessage(content=query)]
    # different threads inside the checkpointer for multiple conversation
    thread = {"configurable": {"thread_id": "1"}}

    async def main():
        async for event in ai_bot.app.astream_events({"messages": messages}, thread, version="v1"):  # type: ignore
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content  # type: ignore
                if content:
                    # Empty conent in the context of OpenAI means that the model is asking for a tool to be invoked. So we only print non-empty content
                    print(content, end="", flush=True)

    asyncio.run(main())
