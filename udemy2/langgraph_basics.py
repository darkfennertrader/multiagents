# %%
from typing import TypedDict, Annotated, List
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from IPython.display import Image, display
from config import set_environment_variables
from langgraph.graph import StateGraph, MessageGraph, MessagesState, END
from langgraph.graph.message import add_messages


set_environment_variables("LangGraph_basics")


model = ChatOpenAI(model="gpt-4o", temperature=0.0)


class State(TypedDict):
    messages: Annotated[List, add_messages]


def bot(state: State):
    print("\nwithin bot function")
    print(state["messages"])
    print("-" * 100)
    response = model.invoke(state["messages"])
    return {"messages": response}


graph_builder = StateGraph(State)
graph_builder.set_entry_point("bot")
graph_builder.add_node("bot", bot)
graph_builder.set_finish_point("bot")
app = graph_builder.compile()


# display(Image(app.get_graph(xray=True).draw_png()))  # type: ignore

# %%

if __name__ == "__main__":

    # msgs1 = [HumanMessage(content="Hello", id="1")]
    # msgs2 = [AIMessage(content="Hi there!", id="2")]
    # print(add_messages(msgs1, msgs2))  # type: ignore

    # message = "Hello, how are you?"
    # # use of invoke
    # res = app.invoke({"messages": [message]})
    # print()
    # print(res["messages"])

    # use of stream
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break

        for event in app.stream(
            {"messages": ("user", user_input)}, stream_mode="values"
        ):
            for value in event.values():
                print("Assistant:", value)
