from typing import List, Union
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from callbacks import AgentCallbackHandler
from config import set_environment_variables

set_environment_variables("React_from_scratch")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    verbose=False,
    # this is necessary to truncate the output from the LLM
    model_kwargs={"stop": ["\nObservation", "Observation"]},
    callbacks=[AgentCallbackHandler()],
)


@tool("get_text_length", return_direct=False)
def get_text_length(text: str) -> int:
    """Returs the length of a text by characters."""
    # stripping away non alphabetical characters if any
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


tools = [get_text_length]


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


react_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


react_prompt = PromptTemplate.from_template(template=react_template).partial(
    tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])  # type: ignore
)

# print(react_prompt)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | react_prompt
    | llm
    | ReActSingleInputOutputParser()
)


intermediate_steps = []
agent_step: Union[AgentAction, AgentFinish] = ""
i = 1
while not isinstance(agent_step, AgentFinish):

    print(f"\nAGENT STEP {i}:")
    print("*" * 15)
    agent_step = agent.invoke(
        {
            "input": "What is the text length of 'Reinforcement Learning' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print("\nOBSERVATION: ", observation)
        intermediate_steps.append((agent_step, str(observation)))

    i += 1

print("\nAGENT FINISH:")
if isinstance(agent_step, AgentFinish):
    print(agent_step.return_values)

# # Fake AgentExecutor impementation: It is a fancy while loop where we keep making LLM calls, then interpreting what the LLM has responded and then choosing the correct tool to use, running it, then running it over and over again until LLM decides to stop.
# class FakeAgentExecutor:
#     def __init__(self, agent, tool_to_run):
#         self.agent = agent
#         self.tool_to_run = tool_to_run

#     def invoke(self, input, tool_input):
#         while True:
#             result = self.agent(input)
#             if result == "RunTool":
#                 self.tool_to_run(tool_input)  # type: ignore
#             else:
#                 return result
