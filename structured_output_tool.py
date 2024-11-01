import operator
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph


VISION_MODEL = "dall-e-3"
openai_client = OpenAI()
model = ChatOpenAI(temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def generate_image(text: str) -> str | None:
    """This tool generates an image based on the text provided.
    The output"""
    # print(size)
    # print(quality)
    # print(style)
    response = openai_client.images.generate(
        model=VISION_MODEL,
        prompt=text,
        size="1024x1024",
        quality="standard",
        style="vivid",
        response_format="url",
    )
    return response.data[0].url


tools = [generate_image]
tool_node = ToolNode(tools)


class Url(BaseModel):
    url: str = Field(
        description="the url of the image that starts with: 'https://....' "
    )


class Response(BaseModel):
    """Final response to the user."""

    type: str = Field(description="it must always be equal to 'image_url'")
    image_url: Url


# Bind to the actual tools + the response format!
model = model.bind_tools(tools + [Response])
# model = model.bind_tools(tools)


# Define the function that determines whether to continue or not {'response': 'Here is an image of a creepy Artificial Intelligence:', 'image_paths'
def route(state: AgentState) -> Literal["action", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise if there is, we need to check what type of function call it is
    if last_message.tool_calls[0]["name"] == Response.__name__:
        return "__end__"
    # Otherwise we continue
    return "action"


# Define the function that calls the model
def call_model(state: AgentState):
    print("\nCALLING LLM:")
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    route,
)
workflow.add_edge("action", "agent")
app = workflow.compile()


if __name__ == "__main__":

    inputs = {"messages": [HumanMessage(content="generate a blu cat image")]}
    for output in app.stream(inputs, stream_mode="values"):
        last_msg = output["messages"][-1]
        print(output["messages"])
        print(last_msg)
        print("\n---\n")
