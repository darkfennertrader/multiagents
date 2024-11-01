# %%
import operator
from typing import TypedDict, Annotated
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
from langgraph.graph import MessageGraph, StateGraph, END

from config import set_environment_variables

set_environment_variables("react_from_basics")

tool = TavilySearchResults(max_results=2)
# print(tool.name, type(tool))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system="") -> None:
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exist_action, {True: "action", False: END}  # type: ignore
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.app = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        print("message from openai:")
        print(message)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        print("\nTAKE ACTION:")
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        print(tool_calls)
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
    ai_bot = Agent(model, [tool], system=prompt)
    display(Image(ai_bot.app.get_graph().draw_png()))  # type: ignore

    # %%

    query = "What is the weather in Milan and Rome?"
    # query = "Who won the SuperBowl in 2024? What is the GDP of that state?"

    messages = [HumanMessage(content=query)]
    result = ai_bot.app.invoke({"messages": messages})
    print()
    print(result["messages"][-1].content)
