# %%
import re
from typing import Annotated, TypedDict, Sequence, List, Dict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from rewoo_prompts import PLANNER_PROMPT, SOLVER_PROMPT
from langgraph.graph import StateGraph, END
from IPython.display import Image

from config import set_environment_variables

set_environment_variables("Multi_Agent_ReWOO")

SOLCCER_EXPER_AGENT_NAME = "Davide"
SOLVER_AGENT_NAME = "solver_agent"


TAVILY_TOOL = TavilySearchResults()
LLM = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.5)


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: Dict
    result: str


task = """
You are tasked to write a role-playing game between a chatbot who acts like a Client and the User who pretends to be a SalesPerson. The output should be composed of the following parts:

1) Five of the followings human features (choose them randomly):
    a) Awareness: confidence in one's abilities, decisiveness.
    b) Motivation: energy and passion for what you do
    c) Determination: ability to achieve set goals
    d) Flexibility: adaptability
    e) Communication: ability to clearly convey information and listen
    f) Sociability: ability to build relationships with others
    g) Resilience: ability to cope with stress
    h) Emotional intelligence: managing emotions
    i) Proactivity: taking personal responsibility for what happens
    j) Organization: ability to perform activities in an orderly and organized manner
    k) Productivity: output/performance
    l) Dynamism: interest/curiosity
    m) Cooperativity: ability to work in a team
    n) Leadership: ability to lead workgroups
    
2) A compelling title for the story

3) An interesting and coherent story that should be followed by the chatbot during the session based on the above points that contains 5 challenging open questions, one for each of the above human features chosen at random.

This is an example of the output you must produce:

1) Human features chosen at random:
Awareness, Motivation, Determination, Flexibility and Communication

2) A compelling title for the story:
The Unexpected Turnaround

3) An interesting and coherent story:
Once upon a time, in a bustling city filled with skyscraping corporations and competitive markets, there was a skilled salesperson named Alex. Alex worked for a renowned tech company that had just launched an innovative software solution. The company was eager to secure a major deal with a prestigious client, Titan Industries, a deal that could skyrocket Alex's career and significantly boost the company's revenue.Alex prepared thoroughly, learning everything about Titan Industries and its CEO, Mr. Jonathan Hayes. Alex walked into the meeting room with a confident smile, carrying a presentation that had taken days to perfect. As they sat down, Mr. Hayes skipped the small talk. "I've seen your initial pitch, Alex. Before we dive into specifics, I have a few concerns," he said in a firm tone. The atmosphere grew tense. Alex knew this wasn't going to be an easy conversation.

**1. Awareness**

Mr. Hayes looked directly at Alex. "Are you aware of the latest changes in our industry? How does your software account for the dynamic shifts we're experiencing, especially considering the new regulations that just came into effect?"

Alex felt the challenge right away. To answer this, Alex needed to demonstrate a deep knowledge of the industry's evolving landscape and how the software could adapt.

**2. Motivation**

Mr. Hayes leaned back in his chair, his eyes narrowing just slightly. "What motivates you, Alex? Is it meeting your sales targets, or do you have a genuine desire to solve the issues that companies like ours face?"

Alex had to show personal motivation beyond quotas and commissions, revealing a genuine passion for the client's success.

**3. Determination**

“We've faced countless disappointments with other software providers before. What makes you so certain that your solution will not only meet but exceed our expectations this time? How determined are you to ensure that this isn't just another failed investment for us?" Mr. Hayes asked, almost cynically.

Alex knew that this question demanded more than just rehearsed answers. It required a demonstration of unwavering commitment and a strong assurance of follow-through.

**4. Flexibility**

Just as Alex thought things couldn't get more complex, Mr. Hayes threw another curveball. “Suppose our main requirements change halfway through implementation due to unforeseen circumstances. How flexible is your approach? Can you adapt swiftly without compromising the quality or timeline?”

Flexibility in handling such scenarios was crucial, and Alex needed to convince Mr. Hayes that the software, and the team behind it, were agile enough to accommodate shifts in requirements.

**5. Communication**

Finally, Mr. Hayes asked a question that aimed straight at Alex's ability to maintain ongoing rapport. “In your experience, what is your communication strategy in ensuring all stakeholders are kept informed and aligned throughout the project? Can you give an example where this ability significantly influenced the project's success?”

Alex realized that clear, transparent communication could make or break the project. Providing a tangible example would be key in instilling confidence.

The questions hung heavily in the air, and Alex knew that answering them convincingly would be the ultimate test of skills. Taking a deep breath, they began to weave a compelling narrative that addressed each concern with confidence and clarity, hoping to turn the challenging situation into an opportunity for unprecedented success.

---

**For Sales Training:**

- **Awareness:** Are the salespeople well-versed in the latest industry trends and regulations?
- **Motivation:** Do they have a genuine passion for helping clients succeed?
- **Determination:** Can they demonstrate a commitment to overcoming challenges and ensuring client satisfaction?
- **Flexibility:** Are they capable of adapting to changing client needs without losing momentum?
- **Communication:** Do they have strategies in place for maintaining clear and effective communication with all stakeholders?

Each of these questions serves not just to test knowledge or capability but to gauge the deeper qualities that make a salesperson truly exceptional.


"""

# result = LLM.invoke(PLANNER_PROMPT.format(task=task))
# regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
# matches = re.findall(regex_pattern, result.content)
# print("\nPLAN_STRING:")
# print(result.content)
# print("\nMATCHES:")
# print(matches)


planner_prompt_template = ChatPromptTemplate.from_messages([("user", PLANNER_PROMPT)])
planner_chain = planner_prompt_template | LLM


def get_plan(state: ReWOO):
    task = state["task"]
    regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
    result = planner_chain.invoke({"task": task})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)  # type: ignore
    return {"steps": matches, "plan_string": result.content}


def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]  # type: ignore
    _results = state["results"] or {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        result = TAVILY_TOOL.invoke(tool_input)
    elif tool == "LLM":
        result = LLM.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = SOLVER_PROMPT.format(plan=plan, task=state["task"])
    result = LLM.invoke(prompt)
    return {"result": result.content}


def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"


graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

app = graph.compile()

Image(app.get_graph().draw_png())  # type: ignore

# %%


for s in app.stream({"task": task}):
    print(s)
    print("---")
