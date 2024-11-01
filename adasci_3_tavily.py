# %%
import asyncio
import json
from time import sleep, time
import operator
from typing import Annotated, TypedDict, List, Sequence, Literal
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph, StateGraph, END
from langgraph.graph.message import add_messages
from config import set_environment_variables

set_environment_variables("langgraph_basics")

memory = SqliteSaver.from_conn_string(":memory:")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tool = TavilySearchResults(max_results=2)
tools = [tool]
# tool.invoke("What is LangGraph?")
ll_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    count: int
    messages: Annotated[Sequence[BaseMessage], operator.add]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps((tool_result)),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])


def route_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise route to the end."""

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Steve Jobs"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chatbot_chain = chatbot_prompt | ll_with_tools


def chatbot(state: AgentState):
    print("STATE:")
    print(state)
    print("-" * 50)

    # message = chatbot_chain.invoke(state["messages"])
    message = chatbot_chain.invoke({"messages": state["messages"]})
    return {"messages": [message]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot", route_tools, {"tools": "tools", "__end__": "__end__"}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
app = graph_builder.compile(checkpointer=memory)

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
        thread = {"configurable": {"thread_id": "1"}}
        messages = [
            HumanMessage(content=user_input),
        ]
        for event in app.stream(
            input={"messages": messages, "count": 2}, config=thread  # type: ignore
        ):
            for value in event.values():
                print(value)
                # print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    main()
