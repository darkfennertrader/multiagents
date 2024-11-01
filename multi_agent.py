import json
from typing import Annotated, List, Sequence, TypedDict
from time import time
import operator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from elevenlabs import play, stream, save
from elevenlabs.client import ElevenLabs
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langgraph.graph import END, MessageGraph, StateGraph


from config import set_environment_variables

set_environment_variables("Speech")


client = ElevenLabs()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tavily_tool = TavilySearchResults(max_results=3)
tools = [TavilySearchResults(max_results=1)]
tool_executor = ToolExecutor(tools)
llm = llm.bind_tools(tools)


def text_to_speech(state):
    """
    Use this tools to transform text to speech
    """
    text = state["message"][-1].content
    audio_stream = client.generate(
        text=text,
        voice="Raimondo Marino",  # "Davide"
        model="eleven_multilingual_v2",
        stream=True,
    )
    return stream(audio_stream)


# print(text_to_speech.args)
# text_to_speech.run({"text": "   Hello how are you doing?", "voice": "Davide"})


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # sender: str


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


def last_model(state):
    human_input = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "text_to_speech",
                        "args": {
                            "text": human_input,
                        },
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


builder = StateGraph(AgentState)

builder.add_node(key="agent", action=call_model)
builder.add_node(key="call_tool", action=call_tool)
builder.add_node(key="text_to_speech", action=text_to_speech)
builder.set_entry_point("agent")
builder.add_conditional_edges(
    start_key="call_tool",
    condition=should_continue,
    conditional_edge_mapping={
        "continue": "call_tool",
        "end": END,
    },
)
builder.add_edge("call_tool", "agent")

graph = builder.compile()


if __name__ == "__main__":

    start = time()

    inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    for output in graph.stream(inputs):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
    print("\n---\n")

    print(f"\nIt took {(time() -start):.2f} sec. to evaluate your request")
