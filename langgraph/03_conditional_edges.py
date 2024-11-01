# %%
from typing import Literal, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from IPython.display import Image
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, MessageGraph
from config import set_environment_variables

set_environment_variables("LangGraph_base")


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


model = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
model_with_tools = model.bind_tools([multiply])
tool_node = ToolNode([multiply])


def call_oracle(messages: list):
    return model.invoke(messages)


graph = MessageGraph()

graph.add_node("oracle", model_with_tools)
graph.add_node("multiply", tool_node)
graph.add_edge("multiply", END)

graph.set_entry_point("oracle")


def router(state: List[BaseMessage]) -> Literal["multiply", "__end__"]:
    print("\nWithin ROUTER:")
    print(state)
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    print("\ntool_calls arg:")
    print(tool_calls)
    if len(tool_calls):
        return "multiply"
    else:
        return "__end__"


graph.add_conditional_edges("oracle", router)

app = graph.compile()

Image(app.get_graph().draw_png())  # type: ignore
# %%

if __name__ == "__main__":
    resp = app.invoke(HumanMessage("What is 123 * 456?"))  # type: ignore
    print("\nRESPONSE:")
    print(resp)
    print(resp[-1].content)  # type: ignore
    print(type(resp[-1]))  # type: ignore
