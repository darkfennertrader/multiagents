# %%
import json
import operator
from typing import Annotated, TypedDict, Sequence, List, Dict, Optional, Literal
from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AnyMessage
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from config import set_environment_variables

set_environment_variables("SelfDiscoverAgent")


# Function to generate a concise report
def generate_report(data):
    report = []

    for step, description in data.items():
        report.append(f"{step}\n" + "-" * len(step))
        report.append(description)
        report.append("")  # Add a newline for separation between steps

    return "\n".join(report)


reasoning_modules = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    # "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternat3ive perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressfrom langchain_community.tools.tavily_search import TavilySearchResultsed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    # "38. Let’s think step by step."
    "39. Let’s make a step by step plan and implement it with good notation and explanation.",
]


SELECT_PROMPT = """
Select several reasoning modules that are crucial to utilize in order to solve the given task:

All reasoning module descriptions:
{reasoning_modules}

Task: {task_description}

Select several modules are crucial for solving the task above.
YOU MUST ONLY SELECT THE MODULES AND THEIR RELATIVE DESCRIPTION.
"""


ADAPT_PROMPT = """
Rephrase and specify each reasoning module so that it better helps solving the task:

SELECTED module descriptions:
{selected_modules}

Task: {task_description}

Adapt each reasoning module description to better solve the task

"""

STRUCTURED_PROMPT = """
Operationalize the reasoning modules into a step-by-step reasoning plan:

Here's an example:

Example task:

If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:

"Position after instruction 1":
"Position after instruction 2":
"Position after instruction n":
"Is final position the same as starting position":


Adapted module description:
{adapted_modules}

Task: {task_description}

Implement a reasoning structure for solvers to follow step-by-step and arrive at correct answer.

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive at the correct conclusion for tasks like this.

"""

REASONING_PROMPT = """
Follow the step-by-step reasoning plan to correctly solve the task. Fill in the values following the keys by reasoning specifically about the task given. Do not simply rephrase the keys.
    
Reasoning Structure:
{reasoning_structure}

Task: {task_description}

"""


model = ChatOpenAI(model="gpt-4o", temperature=0)
TAVILY_TOOL = TavilySearchResults(max_results=3)
tools = [TAVILY_TOOL]


class SelfDiscoverState(TypedDict):
    task_description: str
    reasoning_modules: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]


model = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")


def select(state: SelfDiscoverState):
    print("\nSELECT")
    select_prompt = SELECT_PROMPT.format(
        reasoning_modules=state["reasoning_modules"],
        task_description=state["task_description"],
    )
    response = model.invoke(select_prompt).content
    # print(response)
    return {"selected_modules": response}


def adapt(state: SelfDiscoverState):
    print("\nADAPT")
    adapt_prompt = ADAPT_PROMPT.format(
        selected_modules=state["selected_modules"],
        task_description=state["task_description"],
    )
    response = model.invoke(adapt_prompt).content
    return {"adapted_modules": response}


def structure(state: SelfDiscoverState):
    structured_prompt = STRUCTURED_PROMPT.format(
        adapted_modules=state["adapted_modules"],
        task_description=state["task_description"],
    )
    response = model.invoke(structured_prompt).content

    return {"reasoning_structure": response}


def reason(state: SelfDiscoverState):
    print("\nREASON")
    reasoning_prompt = REASONING_PROMPT.format(
        reasoning_structure=state["reasoning_structure"],
        task_description=state["task_description"],
    )
    response = model.invoke(reasoning_prompt).content
    return {"answer": response}


graph = StateGraph(SelfDiscoverState)
graph.add_node(select)
graph.add_node(adapt)
graph.add_node(structure)
graph.add_node(reason)
graph.add_edge(START, "select")
graph.add_edge("select", "adapt")
graph.add_edge("adapt", "structure")
graph.add_edge("structure", "reason")
graph.add_edge("reason", END)
app = graph.compile()

# This works only in jupyter notebok integrated into VSCode
# display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %%


task = """
I would like to write a report structure based on this process: The business employee makes a self-evaluation on 14 soft skills (ranking them on a scale between 1 to 10) than takes an objective test based on the same soft skills that produces an objective ranking. Comparing the self-evaluation and the objective test a gap analysis is produced that highlights differences between the two tests on some of the factors (overestimation or underestimation). Baesd on these identified gaps, the employees takes some training courses, each of one addresses a single gap (e.g. train in leadership, determination, ecc..). The single training course is composed of a certain number of sessions. A sessione puts the employee in front of a use case to solve by answering the question posed by the trainer. At the end of the session a feedback is given to the user based on the responses of the session. When the employee terminates all courses a report is generated. Your objective is to help me delineate on how the report should be written in terms of structure, paragraphs... So Write the skeleton that should be used for writing the report. The filling of the paragraphs is not your concerns. Just the skeleton and what each paragraph should contain
"""

reasoning_modules_str = "\n".join(reasoning_modules)


for s in app.stream(
    {"task_description": task, "reasoning_modules": reasoning_modules_str}
):
    for key, value in s.items():
        print(f"\nRESPONSE FROM NODE: {key}")
        if key == "reason":
            print(value["answer"])
        else:
            print(value)
