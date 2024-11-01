# %%
import random
from typing import Annotated, TypedDict, Union
import operator
import langchain
from langchain import hub
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import StateGraph, END
from IPython.display import Image

from config import set_environment_variables

# langchain.debug = True

set_environment_variables("Youtube - AgentExecutor")

# get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@tool("lower_case", return_direct=True)
def to_lower_case(input: str) -> str:
    """Returns the input as all lower case."""
    return input.lower()


@tool("random_number", return_direct=True)
def random_number_maker(input: str) -> str:
    """Returns a random number between 0-100."""
    return str(random.randint(0, 100))


tools = [to_lower_case, random_number_maker]

tool_executor = ToolExecutor(tools)

# agent_runnable = create_openai_tools_agent(llm, tools, prompt)

agent_runnable = create_openai_functions_agent(llm, tools, prompt)

# print("\nPROMPT")
# print(prompt.get_prompts())

inputs = {
    "input": "give me a random number and then write in words and make it lower case.",
    "chat_history": [],
    "intermediate_steps": [],
}

agent_outcome = agent_runnable.invoke(inputs)
# print("\nAGENT OUTCOME:")
# print(agent_outcome)
# print(type(agent_runnable))
# print(type(agent_outcome))


# define agent/graph
def run_agent(data):
    print("\nEXECUTE AGENT NODE:")
    agent_outcome = agent_runnable.invoke(data)
    print("\nAgent outcome:")
    print(agent_outcome)
    print("*" * 40)
    return {"agent_outcome": agent_outcome}


# define the function to execute tools
def execute_tools(data):
    print("\nEXECUTE TOOLS FUNCTION:")
    # Get the most recent agent_outcome - this is the key added in the agent above
    agent_action = data["agent_outcome"]
    # Execute the tool
    output = tool_executor.invoke(agent_action)
    print(f"The agent action is \n{agent_action}")
    print(f"The tool result is \n{output}")
    print("*" * 40)
    # Return the output
    return {"intermediate_steps": [(agent_action, str(output))]}


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")
app = workflow.compile()

# Image(app.get_graph().draw_png())  # type: ignore

# %%


inputs = {
    "input": "give me a random number and then write in words and make it lower case.",
    "chat_history": [],
}
# for s in app.stream(inputs):
#     print(list(s.values())[0])
#     print("----")

output = app.invoke(inputs)
print(output.get("agent_outcome").return_values["output"])  # type: ignore
print()
print(output.get("intermediate_steps"))
