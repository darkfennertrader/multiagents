# %%
from typing import Annotated, TypedDict, Union, Tuple, List
import operator
from IPython.display import Image, display
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from config import set_environment_variables

set_environment_variables("LangGraph: React_Agent")

react_prompt: PromptTemplate = hub.pull("hwchase17/react")
print(react_prompt.template)


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    return: the number tripled - > multiplied by 3
    """
    return 3 * float(num)


tools = [TavilySearchResults(max_results=3), triple]
tools = [TavilySearchResults(max_results=3)]
llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
react_agent_runnable = create_react_agent(llm, tools, react_prompt)
tool_executor = ToolExecutor(tools)


class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]


def run_agent_reasoning_engine(state: AgentState):
    print("\nLLM:")
    agent_outcome = react_agent_runnable.invoke(state)
    print(agent_outcome)
    return {"agent_outcome": agent_outcome}


def execute_tools(state: AgentState):
    print("\nTool:")
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    print(output)
    return {"intermediate_steps": [(agent_action, str(output))]}


AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


flow = StateGraph(AgentState)
flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)
flow.add_conditional_edges(AGENT_REASON, should_continue)
flow.add_edge(ACT, AGENT_REASON)
app = flow.compile()

# # This works only in jupyter notebok integrated into VSCode
Image(app.get_graph().draw_png())  # type: ignore
#  %%

if __name__ == "__main__":

    _input = """
    Dato il seguente 'CONCETTO' fornisci una descrizione approfondita considerando che dovra' essere usato per creare un 'GIOCO' per allenare le persone a migliorarsi su di esso.
    \n\nGIOCO: la persona interpreta un ruolo tipico di una situazione aziendale. L'interlocutore e' un chatbot che utilizzando la storia come base fara' domande allo user sul 'CONCETTO' in modo da far raggiungere uno degli 'OBIETTIVI' di miglioramento.
    
    \n\nOBIETTIVI:
    \nSOPRAVVALUTAZIONE: l'utente si sopravaluta sul 'CONCETTO' ovvero si considera piu' abile rispetto alla sua valutazioine oggettiva.
    
    \nSOTTOVALUTAZIONE:l'utente si sottovaluta sul 'CONCETTO' ovvero si considera meno abile rispetto alla sua valutazione oggettiva.
    
    \nBASSI VALORI: l'utente aha dei valori oggettivi sul 'CONCETTO' bassi e deve migiorare
    
    
    \n\nCONCETTO: Collaboratività
    Il fattore collaboratività è molto diverso e da non confondere con quello di socievolezza.
    La collaboratività è il saper lavorare in team e sentirsi parte di un team.
    La collaboratività alta indica essere che la persona è un gregario, un follower. Non c’è la tendenza a guidare un gruppo di lavoro bensì a seguire coloro che danno direttive. Il collaboratore ideale è colui che possiede alti valori in questo fattore. Non ha l’ambizione di prevaricare gli altri poiché per lui/lei il fine principale è il bene del gruppo. Si sente bene quando ne fa parte e l’unità è ciò che considera l’obiettivo più importante.
    Alcune frasi che distinguono alta collaboratività sono le seguenti: “il gruppo è importante”; “per il gruppo ha senso”, “lo faccio per il gruppo”
    Al contrario persone con collaboratività bassa non hanno atteggiamenti particolarmente collaborativi. Lavorano bene da soli e tendenzialmente sono individualisti. Il gruppo può essere funzionale per raggiungere gli obiettivi.


    Utilizza il seguente SCHEMA per l'output:
    \n\nSCHEMA
    
    \nDESCRIZIONE CONCETTO (usa almeno 200 parole per la descrizione del concetto in modo approfondito)
    
    \n\nSOPRAVALUTAZIONE
    \nDescrizione
    \nComportamento
    \nDescrizione sfide e meccaniche del gioco
    \nCriteri di valutazione
    
    \n\nSOTTOVALUTAZIONE
    \nDescrizione
    \nComportamento
    \nDescrizione sfide e meccaniche del gioco
    \nCriteri di valutazione

    \n\nBASSI LIVELLI
    \nDescrizione
    \nComportamento
    \nDescrizione sfide e meccaniche del gioco
    \nCriteri di valutazione
    
    \n\nCONCLUSIONI
    
    \nRISPONDI SEMPRE IN ITALIANO
    """

    # Image(app.get_graph().draw_png())  # type: ignore
    # plt.show()

    response = app.invoke(
        input={
            "input": _input,
            "intermediate_steps": [],
        }
    )
    print(response)
