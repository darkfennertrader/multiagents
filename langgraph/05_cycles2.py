# %%
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from IPython.display import Image
from langgraph.graph import END, MessageGraph


model = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)


def entry(input: List[HumanMessage]):
    return input


def action(input: List[HumanMessage]):
    print("Action taken:", [msg.content for msg in input])
    if len(input) > 5:
        input.append(HumanMessage(content="end"))
    else:
        input.append(HumanMessage(content="continue"))


def should_continue(input: List[HumanMessage]):
    last_message = input[-1]
    if "end" in last_message.content:
        return "__end__"
    return "action"


workflow = MessageGraph()

# Define the two nodes we will cycle between
workflow.add_node("agent", entry)
workflow.add_node("action", action)
workflow.add_edge("action", "agent")

workflow.set_entry_point("agent")


# We now add a conditional edge
workflow.add_conditional_edges(
    "agent", should_continue, {"action": "action", "__end__": END}
)

app = workflow.compile()

Image(app.get_graph().draw_png())  # type: ignore
# %%

if __name__ == "__main__":

    # for output in app.stream(HumanMessage(content="Hello")):
    #     # stream() yields dictionaries with output keyed by node name
    #     for key, value in output.items():
    #         print(f"Output from node '{key}':")
    #         print("---")
    #         print(value)
    #     print("\n---\n")
    app.invoke(HumanMessage(content="Hello"))

# %%
