# %%
import operator
import uuid
from typing import TypedDict, Annotated, List, Dict, Literal
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessageGraph, StateGraph, START, END

from config import set_environment_variables

set_environment_variables("prompt_generator")

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


SYSTEM_PROMPT = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chatbot_chain = chatbot_prompt | llm

# tool = TavilySearchResults(max_results=2)
# print(tool.name, type(tool))


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


llm = ChatOpenAI(temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])


def get_messages_info(messages):
    return [SystemMessage(content=SYSTEM_PROMPT)] + messages


chain = get_messages_info | llm_with_tool


# def llm_node(state: AgentState):
#     print(state)
#     messages = state["messages"]
#     response = chain.invoke(messages)  # type: ignore
#     print("after:")
#     print(response)

# return {"messages": [response]}


# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


prompt_gen_chain = get_prompt_messages | llm


def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


memory = SqliteSaver.from_conn_string(":memory:")
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: list):
    return ToolMessage(
        content="Prompt generated!", tool_call_id=state[-1].tool_calls[0]["id"]
    )


workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
app = workflow.compile(checkpointer=memory)

display(Image(app.get_graph().draw_png()))  # type: ignore

# %%

if __name__ == "__main__":

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        user = input("User (q/Q to quit): ")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        output = None
        for output in app.stream(
            [HumanMessage(content=user)], config=config, stream_mode="updates"  # type: ignore
        ):
            last_message = next(iter(output.values()))
            last_message.pretty_print()

        if output and "prompt" in output:
            print("Done!")
