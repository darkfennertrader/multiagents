# %%
from time import time
from typing import List
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from parallel_tool_executor import execute_tools
from langgraph.graph import START, END, MessageGraph
from chains import first_responder, revisor

from schemas import AnswerQuestion

from config import set_environment_variables

set_environment_variables("Reflexion")

MAX_ITERATIONS = 3

llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
# Transform the function call returned from the LLM into a dict
parser = JsonOutputToolsParser(return_id=True)
# It's going to search for the function calling invocation, parse it and transform it into an AnswerQuestion object
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revisor", revisor)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revisor")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    else:
        return "execute_tools"


builder.add_conditional_edges(
    "revisor", event_loop, {"execute_tools": "execute_tools", END: END}
)
builder.set_entry_point("draft")
app = builder.compile()

# # This works only in jupyter notebok integrated into VSCode
# Image(app.get_graph().draw_png())  # type: ignore
# Image(app.get_graph().draw_png())  # type: ignore
display(Image(app.get_graph().draw_mermaid_png(output_file_path="reflexion.png")))  # type: ignore

# %%

if __name__ == "__main__":

    message = """
    Data la seguente diagnosi medica spiegare in modo elementare frase per frase.
    \n\nNote: esame EMG eseguito in condizioni di parziale compliance. Muscoli esaminati:
    - glosso sinistro: attivita' spontanea assente per quanto valutabile, fasi durata ampiezza nei limiti per quanto valutabile, transizione ricco.
    - bicipite destro: attivita' spontanea assente, fase durata ampiezza lievemente aumentata, transizione ricco.
    - I interosseo destro e sinistro: fascicolazione 2+, fasi durata ampiezza aumentate, transizione povero.
    - vasto mediale destro e sinistro: fascicolazione 1+ attivita' volontaria non valutabile.
    - tibiale anteriore destro: firbillazione 1+, fasi durata ampiezza aumentate, transizione ricco.
    - tibiale anteriore sinistro: attivita' spontanea assente, fasi durata ampiezza aumentate, transizione ricco.
    \nConclusioni: reperti neurofisiopatologici compatibili con il sospetto clinico di malattia del motoneurone.

    Dai inoltre un tuo parere medico su cosa dovrebbe essere fatto per confermare o meno la malattia del motoneurone.
    Rispondi in italiano.
    
    """

    start = time()
    # res = app.invoke(message)
    # print(res[-1].tool_calls[0]["args"]["answer"])  # type: ignore
    # print()
    # print(res[-1].tool_calls[0]["args"]["references"])  # type: ignore

    for event in app.stream(message, stream_mode="updates"):  # type: ignore
        print()
        print(event)  # type: ignore
        print("-" * 80)

    print(f"\nIt took {(time() -start):.2f} sec. to evaluate your request")
