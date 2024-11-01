# %%
import asyncio
from time import sleep, time
import operator
from typing import Annotated, TypedDict, List, Sequence
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    HumanMessage,
    AnyMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from elevenlabs import play, stream, save, Voice, VoiceSettings
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import MessageGraph, StateGraph, END
from langgraph.graph.message import add_messages
from config import set_environment_variables

set_environment_variables("langgraph_basics")

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    count: int
    messages: Annotated[List[BaseMessage], operator.add]


graph_builder = StateGraph(AgentState)


def chatbot(state: AgentState):
    print(state["count"])
    message = llm.invoke(state["messages"])
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
app = graph_builder.compile()

display(Image(app.get_graph().draw_png()))  # type: ignore

# %%


def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # #####    INPUT TYPE 1   #####
        # # message type must be one of: 'human', 'user', 'ai', 'assistant', or 'system'
        # for event in app.stream(
        #     {
        #         "messages": [
        #             ("system", "You are Steve Jobs"),
        #             ("user", user_input),
        #         ],
        #         "count": 2,
        #     }
        # ):
        #     for value in event.values():
        #         print("Assistant:", value["messages"][-1].content)

        ####    INPUT TYPE 2   #####
        messages = [
            SystemMessage(content="You are Steve Jobs"),
            HumanMessage(content=user_input),
        ]
        for event in app.stream({"messages": messages, "count": 2}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    main()