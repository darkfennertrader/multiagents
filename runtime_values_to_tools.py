# %%
import operator
from typing import List
from typing import Annotated, Sequence, TypedDict
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph


# A global dict that the tools will be updating in this example.
user_to_pets = {}


def generate_tools_for_user(user_id: str) -> List[BaseTool]:
    """Generate a set of tools that have a user id associated with them."""

    @tool
    def update_favorite_pets(pets: List[str]) -> None:
        """Add the list of favorite pets."""
        user_to_pets[user_id] = pets

    @tool
    def delete_favorite_pets() -> None:
        """Delete the list of favorite pets."""
        if user_id in user_to_pets:
            del user_to_pets[user_id]

    @tool
    def list_favorite_pets() -> None:
        """List favorite pets if any."""
        return user_to_pets.get(user_id, [])

    return [update_favorite_pets, delete_favorite_pets, list_favorite_pets]  # type: ignore


model = ChatOpenAI(temperature=0, streaming=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state, config):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    tools = generate_tools_for_user(config["user_id"])
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state, config):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    # We call the tool_executor and get back a response
    # We can now wrap these tools in a simple ToolExecutor.
    # This is a real simple class that takes in a ToolInvocation and calls that tool, returning the output.
    # A ToolInvocation is any class with `tool` and `tool_input` attribute.
    tools = generate_tools_for_user(config["user_id"])
    tool_executor = ToolExecutor(tools)
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    # We use the response to create tool messages
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # We return a list, because this will get added to the existing list
    return {"messages": tool_messages}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)  # type: ignore
workflow.add_node("action", call_tool)  # type: ignore
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()


# try:
#     display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
# %%

if __name__ == "__main__":
    user_to_pets.clear()  # Clear the state
    print("-" * 80)
    print(f"User information prior to run: {user_to_pets}")

    inputs = {"messages": [HumanMessage(content="my favorite pets are cats and dogs")]}
    for output in app.stream(inputs, {"user_id": "eugene"}):  # type: ignore
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    print(f"User information prior to run: {user_to_pets}")

    print("-" * 80)
    inputs = {"messages": [HumanMessage(content="what are my favorite pets?")]}
    for output in app.stream(inputs, {"user_id": "eugene"}):  # type: ignore
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    print(f"User information prior to run: {user_to_pets}")

    print("-" * 80)
    inputs = {
        "messages": [
            HumanMessage(
                content="please forget what i told you about my favorite animals"
            )
        ]
    }
    for output in app.stream(inputs, {"user_id": "eugene"}):  # type: ignore
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    print(f"User information prior to run: {user_to_pets}")
