# %%

import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, MessageGraph
from pydantic.v1 import BaseModel

# %%

# model = ChatOpenAI(temperature=0)

# graph = MessageGraph()

# graph.add_node("oracle", model)
# graph.add_edge("oracle", END)

# graph.set_entry_point("oracle")

# runnable = graph.compile()


# resp = runnable.invoke(HumanMessage("What is 1 + 1?"))
# print(resp)
# print()
# print(resp[0])
# print()
# print(resp[1])
# print(len(resp))
# print(type(resp))

# %%


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


model = ChatOpenAI(temperature=0)
model_with_tools = model.bind(tools=[convert_to_openai_tool(multiply)])

graph = MessageGraph()


def invoke_model(state: List[BaseMessage]):
    return model_with_tools.invoke(state)


graph.add_node("oracle", invoke_model)


def invoke_tool(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    multiply_call = None

    for tool_call in tool_calls:
        if tool_call.get("function").get("name") == "multiply":
            multiply_call = tool_call

    if multiply_call is None:
        raise Exception("No adder input found.")

    res = multiply.invoke(json.loads(multiply_call.get("function").get("arguments")))

    return ToolMessage(tool_call_id=multiply_call.get("id"), content=res)


graph.add_node("multiply", invoke_tool)

graph.add_edge("multiply", END)

graph.set_entry_point("oracle")


def router(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "end"


graph.add_conditional_edges(
    "oracle",
    router,
    {
        "multiply": "multiply",
        "end": END,
    },
)

runnable = graph.compile()

print(runnable.invoke(HumanMessage("What is 123 * 456?")))

# %%
