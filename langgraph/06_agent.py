# %%
import operator
import random
from typing import List, TypedDict, Annotated, Sequence, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from IPython.display import Image
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import END, MessageGraph, StateGraph

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


@tool
def fake_weather_api(city: str) -> str:
    """Check the weather in a specified city. The API is available randomly, approximately every third call."""

    if random.randint(1, 3) == 1:
        return "Sunny, 22C"
    else:
        # return "Service temporary unavailable"
        return "Sunny, 22C"


tools = [fake_weather_api]

tool_executor = ToolExecutor(tools)
model_with_tools = model.bind_tools(tools)

tool_mapping = {"fake_weather_api": fake_weather_api}

##############   TOOL IN DETAILS   ########################
# tool_mapping = {"fake_weather_api": fake_weather_api}
# messages = [
#     HumanMessage(
#         content="How will the weather be in Rome today? I would like to eat outside if possibile"
#     )
# ]
# llm_output = model_with_tools.invoke(messages)
# messages.append(llm_output)  # type: ignore
# print(messages)

# tool debugging:
# for tool_call in llm_output.tool_calls:  # type: ignore
#     tool = tool_mapping[tool_call["name"].lower()]
#     tool_output = tool.invoke(tool_call["args"])
#     messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))  # type: ignore

# model_with_tools.invoke(messages)

# %%


class AgentState(TypedDict):
    api_call_count: int = 0  # type: ignore
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_continue(state: AgentState):
    print("STATE: ", state["messages"])
    last_message = state["messages"][-1]

    if not last_message.tool_calls:  # type: ignore
        return "end"
    else:
        return "continue"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response], "api_call_count": state["api_call_count"]}


def call_tool(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_call = last_message.tool_calls[0]  # type: ignore
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    state["api_call_count"] += 1
    print("Tool output: ", tool_output)
    tool_message = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
    return {"messages": [tool_message], "api_call_count": state["api_call_count"]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")
app = workflow.compile()

Image(app.get_graph().draw_png())  # type: ignore
# %%

if __name__ == "__main__":

    system_message = SystemMessage(
        content="You are responsible for answering user questions. You use tools for that. These tools sometimes fail and you are very resilient and trying them again"
    )
    human_message = HumanMessage(content="what is the weather in sf")
    messages = [system_message, human_message]
    for output in app.stream(
        {"messages": messages, "api_call_count": 0}, stream_mode="updates"
    ):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    # result = app.invoke({"messages": messages, "api_call_count": 0})

# %%
