import operator
from typing import Annotated, TypedDict, List, Literal, Dict, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END


VISION_MODEL = "dall-e-3"

openai_client = OpenAI()
st_client = TavilySearchResults(max_results=3)
memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


def generate_tools(
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
    quality: Literal["standard", "hd"],
    style: Literal["vivid", "natural"],
):

    @tool
    def generate_image(text: str) -> str | None:
        """This tool generates an image based on the user prompt"""
        print(size)
        print(quality)
        response = openai_client.images.generate(
            model=VISION_MODEL, prompt=text, size=size, quality=quality, style=style
        )
        return response.data[0].url

    return generate_image


class Agent:

    def __init__(self, _model, _checkpointer, _system="") -> None:
        self.system = _system
        self.model = _model
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)  # type: ignore
        graph.add_node("action", self.take_action)  # type: ignore
        graph.add_conditional_edges(
            "llm", self.exist_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=_checkpointer)
        self.graph = graph.compile()

    def exist_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0  # type: ignore

    def call_openai(self, state: AgentState, config: Dict[str, Any]):
        print("WITHIN MODEL")
        print(config)
        messages = state["messages"]
        print("\n", "*" * 50)
        print(state)
        print("*" * 50)
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        tools = [
            st_client,
            generate_tools(config["size"], config["quality"], config["style"]),
        ]
        model_with_tools = self.model.bind_tools(tools)
        response = model_with_tools.invoke(messages)
        print("from call_openai:")
        print(response)
        return {"messages": [response]}

    def take_action(self, state: AgentState, config: Dict[str, Any]):
        tool_calls = state["messages"][-1].tool_calls  # type: ignore
        tool_invocations = []
        # A ToolInvocation is any class with `tool` and `tool_input` attribute.
        for tool_call in tool_calls:
            action = ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
            )
            tool_invocations.append(action)

        agent_tools = [
            st_client,
            generate_tools(config["size"], config["quality"], config["style"]),
        ]
        # We can now wrap these tools in a simple ToolExecutor.
        tool_executor = ToolExecutor(agent_tools)
        responses = tool_executor.batch(tool_invocations, return_exceptions=True)

        tool_messages = [
            ToolMessage(
                content=str(response),
                name=tc["name"],
                tool_call_id=tc["id"],  # type: ignore
            )
            for tc, response in zip(tool_calls, responses)
        ]

        print("Back to the model")
        return {"messages": tool_messages}


if __name__ == "__main__":

    prompt = """
    You are a smart research assistant. Use the search engine to look up information. \
    You are also capable of generating an image based on the user prompt. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    size = "1792x1024"
    quality = "hd"
    style = "vivid"

    model = ChatOpenAI(model="gpt-4o")
    chatbot = Agent(_model=model, _checkpointer=memory, _system=prompt)

    # display(Image(chatbot.graph.get_graph().draw_png()))  # type: ignore

    # %%

    # different threads inside the checkpointer for multiple conversation
    params = {
        "configurable": {"thread_id": "1"},  # for persistence
        "size": size,
        "quality": quality,
        "style": style,
    }
    query = "What is the weather in Milan and Rome?"
    # query = "generate an image of space shuttle exploring the universe with the Earth in the background"
    # query = "Who won the SuperBowl in 2024? What is the GDP of that state?"
    messages = [HumanMessage(content=query)]

    result = chatbot.graph.invoke({"messages": messages}, config=params)  # type: ignore

    print()
    print(result["messages"][-1].content)
