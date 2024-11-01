# %%
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from IPython.display import display, Image
from langgraph.graph import END, MessageGraph


model = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)


def add_one(input: list[HumanMessage]):
    input[0].content += "a"  # type : ingore
    return input


graph = MessageGraph()

graph.add_node("branch_a", add_one)
graph.add_node("branch_b", add_one)
graph.add_node("branch_c", add_one)
graph.add_node("final_node", add_one)
graph.add_edge("branch_a", "branch_b")
graph.add_edge("branch_a", "branch_c")
graph.add_edge("branch_b", "final_node")
graph.add_edge("branch_c", "final_node")
graph.add_edge("final_node", END)

graph.set_entry_point("branch_a")

app = graph.compile()

display(Image(app.get_graph().draw_png()))  # type: ignore

# %%
if __name__ == "__main__":
    resp = app.invoke(HumanMessage(content="a"))
    print(resp)
    print("\nRESPONSE:")
    print(resp[-1].content)  # type: ignore
    print(type(resp[-1]))  # type: ignore
