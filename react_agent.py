from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

llm = ChatOpenAI(model="gpt-4o", temperature=0.0, verbose=False)
llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
prompt = hub.pull("hwchase17/react")
# print(prompt.template)

math_tool = Tool(
    name="Calculator",
    description="Useful for when you need to answer questions related to Math",
    func=llm_math,
)

tools = [math_tool, TavilySearchResults(max_results=3)]

# print(tools[0].name)
# print(tools[0].description)

# ReAct framework
react_agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# result = agent_executor.invoke({"input": "What is 3.1^2.1"})

# print(result["output"])

# result = agent_executor.invoke({"input": "How old is the universe?"})
# print(result)

# result = agent_executor.invoke(
#     {
#         "input": "If James was 45 years old three years ago, how old will he be in 2 years?"
#     }
# )
# print(result)

_input = """
    Dato il seguente concetto fornisci una descrizione dettagliata considerando che dovra' essere usato per creare un gioco per allenare le persone a migliorarsi su di esso.
    \n\nCONCETTO: Consapevolezza
    \nQuesto fattore indica la sicurezza in sé e nelle proprie capacità. La sicurezza nelle proprie capacità rende la persona sicura nel prendere decisioni dinanzi a delle scelte o a delle alternativa da vagliare. Livelli medi di consapevolezza permettono alla persona di progredire e di tendere al miglioramento continuo. Il mettersi in discussione non è bloccante nella scelta che non riguarda la propria abilità di scegliere bensì la “migliore scelta da agire”. Al contrario una consapevolezza molto alta indica una certa supponenza, incapacità di mettersi in discussione, indica presunzione. Persone che hanno valori molto alti sovente usano frasi del tipo: “ ormai non ho più nulla da imparare” ; “con tutta la mia esperienza sono arrivato” ; “nulla è più nuovo per me”.
    Rispondi sempre in italiano.
"""

result = agent_executor.invoke({"input": _input})
print(result["output"])
