import operator
from typing import Annotated, List, TypedDict, Union, Tuple
from colorama import Fore, Style
from langchain import hub
from langchain.agents import create_openai_functions_agent, create_openai_tools_agent
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from tools import generate_image, get_weather
from config import set_environment_variables

set_environment_variables("LangGraph Basics")

LLM = ChatOpenAI(streaming=True, temperature=0)
TOOLS = [get_weather, generate_image]
PROMPT = hub.pull("hwchase17/openai-functions-agent")
# PROMPT = hub.pull("hwchase17/openai-tools-agent")

# Example of BaseMessages:
# ("system", "You are a helpful AI bot. Your name is {name}"),
# ("human", "Hello how are you doing?"),
# ("ai", "I'm doing well, thanks!"),
# ("human", "{user_input}")


# AgentAction contains: 1) the tool to be called, 2) input arguments
# AgentFinish contains the final response

# Intermediate_steps example:
# [
#   (
#       AgentAction(tool="tool_1", input={arg1: "value1"}),
#       "{API response JSON object...}" # this is the tool output after the tool was called,
#   ),
#   (
#       AgentAction(tool="tool_2", input={arg2: "value2"}),
#       "Path/to/image.png",
#   ),
# ]


class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]


# the function makes an OpenAI compatible agent
runnable_agent: Runnable = create_openai_functions_agent(LLM, TOOLS, PROMPT)
# runnable_agent: Runnable = create_openai_tools_agent(LLM, TOOLS, PROMPT)

inputs = {
    "input": "give me the weather for New York please.",
    "chat_history": [],
    "intermediate_steps": [],
}

# agent_outcome = runnable_agent.invoke(inputs)  # type: ignore
# print(agent_outcome)


def agent_node(_input: AgentState):
    agent_outcome: AgentActionMessageLog = runnable_agent.invoke(_input)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolExecutor(TOOLS)
# print(tool_executor)


def tool_executor_node(_input: AgentState):
    agent_action = _input["agent_outcome"]
    print()
    print(agent_action)
    print()
    output = tool_executor.invoke(agent_action)
    print(f"Executed {agent_action} with output: {output}")
    return {"intermediate_steps": [(agent_action, output)]}


def should_continue(data: AgentState):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.set_entry_point("agent")
workflow.add_edge("tool_executor", "agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tool_executor", "end": END}
)
app = workflow.compile()


def call_weather_app(query: str):
    inputs = {"input": query, "chat_history": [], "intermediate_steps": []}
    output = app.invoke(inputs)
    result = output.get("agent_outcome").return_values["output"]  # type: ignore
    steps = output.get("intermediate_steps")

    print(f"\n{Fore.BLUE}Result: {result}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Steps: {steps}{Style.RESET_ALL}\n")
    return result


if __name__ == "__main__":
    call_weather_app(
        "Give me a visual image displaying the current weather in Milan, Italy"
    )
