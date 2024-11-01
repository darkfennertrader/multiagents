# %%
from time import time
from typing import Sequence, List
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import generate_chain, reflect_chain

# from tool_executor import execute_tools

from config import set_environment_variables

set_environment_variables("Reflection")

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    print("\nWithin GENERATION NODE")
    return generate_chain.invoke({"messages": state})  # type: ignore


def reflection_node(state: Sequence[BaseMessage]):
    print("\nWithin REFLECTION NODE")
    res = reflect_chain.invoke({"messages": state})  # type: ignore
    # we need to fool the LLM that a human is sending the feedback
    return HumanMessage(content=res.content)  # type: ignore


def should_continue(state: Sequence[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)
app = builder.compile()

display(Image(app.get_graph().draw_png()))  # type: ignore
# Image(app.get_graph().draw_mermaid_png())

# %%

if __name__ == "__main__":

    inputs = HumanMessage("""Causal Machine Learning""")  # type: ignore
    start = time()
    # response = app.invoke(inputs)
    # print(response)
    # for event in app.stream({"messages": ("user", inputs)}):
    for event in app.stream(inputs):
        print(event)
        # for value in event.values():
        #     print(value)
        # print("Assistant:", value["messages"][-1].content)

    print(f"\nIt took {(time() -start):.2f} sec. to evaluate your request")
