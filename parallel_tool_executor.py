import json
from typing import List
from collections import defaultdict
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from schemas import AnswerQuestion, Reflection


from config import set_environment_variables

set_environment_variables()

parser = JsonOutputToolsParser(return_id=True)
search = TavilySearchAPIWrapper()  # type:ignore
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
# wrapper for parallel execution of the tool
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    # for message in state:
    #     print()
    #     print(message)

    tool_invocation: AIMessage = state[-1]  # type: ignore
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        print("\nPARSED CALL")
        print(parsed_call)
        print()
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(tool="tavily_search_results_json", tool_input=query)
            )
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)
    # print("\nOutputs:")
    # print(outputs)

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):

        outputs_map[id_][invocation.tool_input] = output

    # print()
    # print(outputs_map)

    # Convert the mapped outputs to ToolMessage object
    tool_messages = []

    for id_, mapped_output in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_)
        )

    # print()
    # print(json.loads(tool_messages[0].content))

    return tool_messages


if __name__ == "__main__":
    human_message = HumanMessage(content="Write about LLM and Reinforcement Learning")
    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "LLM in reinforcement learning case studies",
            "Comparative analysis of LLM and traditional planners in RL",
            "Detailed process of integrating LLM with reinforcement learning",
        ],
        id="call_KpYHichFFELitHFvFhKy1Ra",  # type: ignore
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFELitHFvFhKy1Ra",
                    }
                ],
            ),
        ],
    )
    print(raw_res)
