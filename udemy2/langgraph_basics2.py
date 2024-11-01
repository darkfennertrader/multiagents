# %%
from typing import TypedDict, Annotated, List
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from config import set_environment_variables
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


set_environment_variables("LangGraph_basics")

memory = SqliteSaver.from_conn_string(":memory:")

model = ChatOpenAI(model="gpt-4o", temperature=0.0)
tool = TavilySearchResults(max_results=2)
tools = [tool]
model_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools=[tool])


class State(TypedDict):
    messages: Annotated[List, add_messages]


def bot(state: State):
    print("\nwithin bot function")
    print(state["messages"])
    print("-" * 100)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": response}


graph_builder = StateGraph(State)
graph_builder.set_entry_point("bot")
graph_builder.add_node("bot", bot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)
# graph_builder.set_finish_point("bot")
app = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])


display(Image(app.get_graph(xray=True).draw_png()))  # type: ignore

# %%

if __name__ == "__main__":

    # print(tool.invoke("What's the weathere today in Milan?"))
    # print(model_with_tools.invoke("What's a 'node' in langgraph?"))
    # msgs1 = [HumanMessage(content="Hello", id="1")]
    # msgs2 = [AIMessage(content="Hi there!", id="2")]
    # print(add_messages(msgs1, msgs2))  # type: ignore

    # message = "Hello, how are you?"
    # # use of invoke
    # res = app.invoke({"messages": [message]})
    # print()
    # print(res["messages"])

    config = {
        "configurable": {"thread_id": 1},
    }

    user_input = "I want to learn about causal inference. Could you do find some good paid courses in Python?"
    events = app.stream(
        {"messages": [("user", user_input)]}, config=config, stream_mode="values"  # type: ignore
    )

    for event in events:
        event["messages"][-1].pretty_print()

    # inspect the state
    snapshot = app.get_state(config)  # type: ignore
    print("\n\n")
    print(snapshot)
    next_step = snapshot.next
    print("next_step ==> ", next_step)

    existing_message = snapshot.values["messages"][-1]
    print("\nTools to be called: ", existing_message.tool_calls)

    # continue the conversation passing None to say continue
    for event in app.stream(None, config=config, stream_mode="values"):  # type: ignore
        print("\n", event["messages"][-1].content)

    # # use of stream
    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() in ["quit", "q", "exit"]:
    #         print("Goodbye!")
    #         break
