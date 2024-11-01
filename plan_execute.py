# %%
import operator
import asyncio
from datetime import datetime
from typing import Annotated, List, Tuple, TypedDict, Union, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from config import set_environment_variables

set_environment_variables("plan_execute")


tools = [TavilySearchResults(max_results=3)]


# class PlanExecute(TypedDict):
#     input: str
#     plan: List[str]
#     past_steps: Annotated[List[Tuple], operator.add]
#     response: str


# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )


# planner_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """For the given objective, come up with a simple step by step plan. \
#             This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
#             The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
#             """,
#         ),
#         ("placeholder", "{messages}"),
#     ]
# )
# planner = planner_prompt | ChatOpenAI(
#     model="gpt-4o", temperature=0
# ).with_structured_output(Plan)


# class Response(BaseModel):
#     """Response to user."""

#     response: str


# class Act(BaseModel):
#     """Action to perform."""

#     action: Union[Response, Plan] = Field(
#         description="Action to perform. If you want to respond to user, use Response. "
#         "If you need to further use tools to get the answer, use Plan."
#     )


# replanner_prompt = ChatPromptTemplate.from_template(
#     """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

# Your objective was this:
# {input}

# Your original plan was this:
# {plan}

# You have currently done the follow steps:
# {past_steps}

# Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
# )


# replanner = replanner_prompt | ChatOpenAI(
#     model="gpt-4o", temperature=0
# ).with_structured_output(Act)


# async def execute_step(state: PlanExecute):
#     plan = state["plan"]
#     plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
#     task = plan[0]
#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
#     agent_response = await agent_executor.ainvoke(
#         {"messages": [("user", task_formatted)]}
#     )
#     return {
#         "past_steps": (task, agent_response["messages"][-1].content),
#     }


# async def plan_step(state: PlanExecute):
#     plan = await planner.ainvoke({"messages": [("user", state["input"])]})
#     return {"plan": plan.steps}  # type: ignore


# async def replan_step(state: PlanExecute):
#     output = await replanner.ainvoke(state)  # type: ignore
#     if isinstance(output.action, Response):  # type: ignore
#         return {"response": output.action.response}  # type: ignore
#     else:
#         return {"plan": output.action.steps}  # type: ignore


# def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
#     if "response" in state and state["response"]:
#         return "__end__"
#     else:
#         return "agent"


# workflow = StateGraph(PlanExecute)


# workflow.add_node("planner", plan_step)
# workflow.add_node("agent", execute_step)
# workflow.add_node("replan", replan_step)
# workflow.add_edge(START, "planner")
# workflow.add_edge("planner", "agent")
# workflow.add_edge("agent", "replan")
# workflow.add_conditional_edges(
#     "replan",
#     should_end,
# )

# app = workflow.compile()

# display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %%

if __name__ == "__main__":
    pass

    # user_input = "what is the hometown of the 2024 Australia open winner?"

    # async def run_app(user_input):
    #     config = {"recursion_limit": 50}
    #     inputs = {"input": user_input}
    #     async for event in app.astream(inputs, config=config):  # type: ignore
    #         for k, v in event.items():
    #             if k != "__end__":
    #                 print(v)

    # asyncio.run(run_app(user_input))
