# %%
import operator
from datetime import datetime
from typing import List, Any, Dict, Annotated, TypedDict, Literal, Sequence
import asyncio
from uuid import uuid4
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import START, StateGraph, END

# from IPython.display import Image, display


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    q_and_a: List[Dict[str, str]]


class ReactAgent:
    def __init__(
        self, model, tools: List[Any], checkpointer, output_parser=None, system=""
    ) -> None:
        self.system = system
        self.checkpointer = checkpointer
        graph = StateGraph(AgentState)
        graph.add_node("agent", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "agent", self.exist_action, {"action": "action", "__end__": END}  # type: ignore
        )
        graph.add_edge("action", "agent")
        graph.set_entry_point("agent")
        self.graph = graph
        self.tools = {t.name: t for t in tools}
        # here we add the output parser if necessary
        self.output_parser = output_parser
        if self.output_parser:
            tools = tools + [output_parser]
            self.model = model.bind_tools(tools, tool_choice="any")
        else:
            self.model = model.bind_tools(tools)

    def compile(self):
        return self.graph.compile(checkpointer=self.checkpointer)

    async def call_openai(self, state: AgentState):
        print("\nAGENT CALL:")
        messages = state["messages"]
        # print("STATE['MESSAGES']:")
        # print(messages)
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = await self.model.ainvoke(messages)
        # print(message)
        return {"messages": [message]}

    async def take_action(self, state: AgentState):
        print("\nTOOL EXECUTOR:")
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        results = []
        for t in tool_calls:
            print(f"Calling tool: {t['name']}, args: {t['args']}")
            result = await self.tools[t["name"]].ainvoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))  # type: ignore
            )
        return {"messages": results}

    def exist_action(self, state: AgentState) -> Literal["action", "__end__"]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:  # type: ignore
            return "__end__"
        if self.output_parser:
            if last_message.tool_calls[0]["name"] == self.output_parser.__name__:  # type: ignore
                return "__end__"
        # Otherwise we continue
        return "action"

    # def display_graph(self):
    #     display(Image(self.app.get_graph().draw_mermaid_png()))


def chatbot_setup():
    SYSTEM = f"""You are an helpful assistant.
    \nCURRENT DATE
    {datetime.now().isoformat()}
    """
    memory = MemorySaver()
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.0, streaming=True)
    tavily = TavilySearchResults(max_results=3)
    tools = [tavily]
    react_agent = ReactAgent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system=SYSTEM,
    )
    return react_agent.compile()


# Output without streaming
async def main(input, graph, thread_id=str(uuid4())):
    config = {"configurable": {"thread_id": thread_id}}
    async for output in graph.astream(input, stream_mode="updates", config=config):
        for node_name, output_value in output.items():
            print(f"Output from node: {node_name}")
            print(output_value)
            # print(type(output_value))
            # print(len(output_value))
            print("\n------------------\n")


# Output with LLM token streaming
async def main2(input, graph, thread_id=str(uuid4())):
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream_events(input, version="v1", config=config):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # print()
            # print(event)
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="")

        # elif kind == "on_tool_start":
        #     print("--")
        #     print(
        #         f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
        #     )
        # elif kind == "on_tool_end":
        #     print(f"Done tool: {event['name']}")
        #     print(f"Tool output was: {event['data'].get('output')}")
        #     print("--")

    print()


async def generate_response(input, graph, thread_id=str(uuid4())):
    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(input, config=config)  # type: ignore
    # return {"messages": [response]}
    return response["messages"][-1].content


# config=RunnableConfig(configurable={"thread_id": uuid4()})
# Look here: https://github.com/JoshuaC215/agent-service-toolkit
async def generate_stream(input, graph, thread_id=str(uuid4())):
    config = {"configurable": {"thread_id": thread_id}}
    async for event in graph.astream_events(input, version="v1", config=config):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # print()
            # print(event)
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="")
                yield content


if __name__ == "__main__":
    # With this  AsyncSqlite the loop does not terminate
    # memory = AsyncSqliteSaver.from_conn_string(":memory:")

    SYSTEM = f"""You are an helpful assistant.
    \nCURRENT DATE
    {datetime.now().isoformat()}
    """

    memory = MemorySaver()
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.0, streaming=True)
    tavily = TavilySearchResults(max_results=3)

    class OutputFormatter(BaseModel):
        """Final response to the user."""

        output: List[str] = Field(description="List of strings.")

    user_query = "What's the weather in Milan and Rome in Italy?"
    # user_query = "tell me 10 words about sport"
    input = {"messages": [HumanMessage(content=user_query)]}

    tools = [tavily]

    react_agent = ReactAgent(
        model=llm,
        tools=tools,
        # output_parser=OutputFormatter,
        checkpointer=memory,
        system=SYSTEM,
    )
    graph = react_agent.compile()

    # Output without streaming
    # asyncio.run(main(input, graph))

    # Output with LLM token streaming
    asyncio.run(main2(input, graph))

    # react_agent.display_graph()
