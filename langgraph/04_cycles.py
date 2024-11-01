# %%
import operator
from typing import Literal, List, TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from IPython.display import Image
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, MessageGraph, StateGraph

tools = [TavilySearchResults(max_results=3)]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True, verbose=True)
model_with_tools = model.bind_tools(tools)


def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

Image(app.get_graph().draw_png())  # type: ignore
# %%

if __name__ == "__main__":

    inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    for output in app.stream(inputs, stream_mode="updates"):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

# %%
