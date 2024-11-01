import operator
from typing import Annotated, List, Tuple, TypedDict, Sequence
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent


model = ChatOpenAI(model="gpt-4o")


@tool
def check_weather(location: str, at_time: datetime | None = None) -> str:
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"


tools = [check_weather]

system_prompt = "You are a helpful bot named Fred."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful bot named Fred."),
        ("placeholder", "{messages}"),
        ("user", "Remember, always be polite!"),
    ]
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]


def modify_state_messages(state: AgentState):
    # You can do more complex modifications here
    print("MODIFY STATE MESSAGES:")
    print(prompt.invoke({"messages": state["messages"]}))
    return prompt.invoke({"messages": state["messages"]})


# graph = create_react_agent(model, tools, state_modifier=modify_state_messages)  # type: ignore
graph = create_react_agent(model, tools, state_modifier=prompt)  # type: ignore
inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
