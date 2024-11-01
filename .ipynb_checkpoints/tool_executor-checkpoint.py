from typing import List
from pprint import pprint
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from schemas import AnswerQuestion, Reflection

from config import (
    OPENAI_API_KEY,
    TAVILY_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
)


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    for message in state:
        print()
        print(message)
    tool_invocation: AIMessage = state[-1]  # type: ignore


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
