# %%
import random
import json
from typing import Annotated, TypedDict, Union, Sequence
import operator
from langchain import hub
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_tools_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
from IPython.display import Image

from config import set_environment_variables

# langchain.debug = True

set_environment_variables("Youtube - ChatExecutor")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)


@tool("lower_case", return_direct=True)
def to_lower_case(input: str) -> str:
    """Returns the input as all lower case."""
    return input.lower()


@tool("random_number", return_direct=True)
def random_number_maker(input: str) -> str:
    """Returns a random number between 0-100."""
    return str(random.randint(0, 100))


tools = [to_lower_case, random_number_maker]

tool_executor = ToolExecutor(tools)  # type: ignore

functions = [convert_to_openai_function(t) for t in tools]
llm = llm.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


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
    print("\nCALL TOOL")
    messages = state["messages"]
    print(messages)
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
    print(f"The agent action is {action}")
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    print(f"The tool result is: {response}")
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


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

# inputs = {"input": "give me a random number and then write in words and make it lower case", "chat_history": []}

system_message = SystemMessage(content="you are a helpful assistant")

user_01 = HumanMessage(content="plear write 'Merlion' in lower case")
user_01 = HumanMessage(content="what is a Merlion?")
user_01 = HumanMessage(
    content="give me a random number and then write in words and make it lower case"
)

inputs = {"messages": [system_message, user_01]}

print(app.invoke(inputs))

# %%
