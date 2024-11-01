# %%
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from IPython.display import display, Image
from langgraph.graph import END, MessageGraph


model = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)


def call_oracle(messages: list):
    print("\nWithin CALL_ORACLE:")
    print(messages, "\n")
    return model.invoke(messages)


graph = MessageGraph()

graph.add_node("oracle", call_oracle)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

app = graph.compile()

display(Image(app.get_graph().draw_png()))  # type: ignore

# %%
if __name__ == "__main__":
    resp = app.invoke(HumanMessage(content="What is 1 + 1?"))
    print(resp)
    print("\nRESPONSE:")
    print(resp[-1].content)  # type: ignore
    print(type(resp[-1]))  # type: ignore
