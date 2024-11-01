# %%
from time import time
from typing import Annotated, TypedDict, List, Sequence
import functools
import operator
from IPython.display import Image, display
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
)
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from elevenlabs import play, stream, save, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from langgraph.graph import END, MessageGraph, StateGraph

from config import set_environment_variables

set_environment_variables("Candidate_Interview")

client = ElevenLabs()


def recruiter_text_to_speech(text):
    """
    Use this tools to transform text to speech
    """
    # text = state["message"][-1].content
    audio_stream = client.generate(
        text=text,
        # voice="Davide",  # "Davide"
        model="eleven_multilingual_v2",
        stream=True,
        voice=Voice(
            voice_id="d2rfZk3QRo5vHPDOBYlA",
            settings=VoiceSettings(
                stability=0.6, similarity_boost=0.8, use_speaker_boost=True, style=0.0
            ),
        ),
    )
    return stream(audio_stream)


def candidate_text_to_speech(text):
    """
    Use this tools to transform text to speech
    """
    # text = state["message"][-1].content
    audio_stream = client.generate(
        text=text,
        model="eleven_multilingual_v2",  # "eleven_multilingual_v2"
        stream=True,
        voice=Voice(
            voice_id="CmiOZ78hHwigIq11xTcn",
            settings=VoiceSettings(
                stability=0.6, similarity_boost=0.8, use_speaker_boost=True, style=0.0
            ),
        ),
    )
    return stream(audio_stream)


RECRUITER_NAME = "interviewer"
CANDIDATE_NAME = "candidate"
EVALUATOR_NAME = "evaluator"

MEMBERS = [RECRUITER_NAME, CANDIDATE_NAME]

RECRUITER_LLM = ChatOpenAI(model="gpt-4o", temperature=0.6, verbose=True)
CANDIDATE_LLM = ChatOpenAI(model="gpt-4o", temperature=0.5, verbose=True)
EVALUATOR_LLM = ChatOpenAI(model="gpt-4o", temperature=0.3, verbose=True)
TAVILY_TOOL = TavilySearchResults(max_results=3)


# You have access to a web search function for additional or up-to-date research if needed. Use this tool to make questions that are relevant to your industry and make use of the the most up-to-date techniques in the field of human resources.
# """


RECRUITER_SYSTEM_PROMPT = """
Sei Davide Malaguti il presidente e fondatore di Formula Coach con oltre 30 anni di esperienza alle spalle nel coaching alle imprese. Formula Coach e' un'azienda che opera nel settore del business coaching. Sei incaricato di valutare un candidato per un ruolo di vendita. Il tuo obiettivo è determinare se il candidato possiede le competenze, l'esperienza e l'attitudine necessarie per eccellere in questa posizione. Per raggiungere questo scopo, coinvolgerai il candidato in un'intervista strutturata, concentrandoti sulla sua esperienza di vendita, sulla comprensione dell'industria del business coaching e sulla capacità di raggiungere gli obiettivi di vendita. Valuterai anche le sue capacità comunicative, di risoluzione dei problemi e il suo approccio alla costruzione e al mantenimento delle relazioni con i clienti. Le tue domande dovrebbero essere progettate per ottenere risposte dettagliate che forniscano una visione delle capacità del candidato e della sua idoneità per il ruolo. Rispondi alla risposta fornita dal CANDIDATO se presente, altrimenti fai una domanda alla volta SFIDANTE!. Quando fornisci la domanda devi essere conciso (massimo tre righe) e DEVI MANTENERE SEMPRE UN COLLOQUIO INFORMALE PER FAR SENTIRE A SUO AGIO IL CANDIDATO CHE INTERVISTI. Le domande devono essere sempre in lingua italiana tranne per gli acronimi inglesi.
"""

recruiter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RECRUITER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

recruiter_chain = recruiter_prompt | RECRUITER_LLM


def recruiter_node(state: Sequence[BaseMessage]):
    print("\nRECRUITER")
    # print(state)
    if len(state) >= 9:
        text = "Grazie per il tuo tempo e la tua disponibilita'. Nelle prossime settimane termineremo il processo di selezione e ti faremo sapere. Buona Fortuna!"
        recruiter_text_to_speech(text)
        return AIMessage(content=text)
    response = recruiter_chain.invoke({"messages": state})  # type: ignore
    recruiter_text_to_speech(response.content)
    return response


CANDIDATE_SYSTEM_PROMPT = """
Sei un candidato che si candida per un ruolo di vendita nel settore del business coaching. Il tuo obiettivo è distinguerti dagli altri candidati fornendo risposte convincenti e perspicaci alle domande poste dal datore di lavoro. Sfrutta la tua esperienza precedente, la comprensione del settore e la tua filosofia personale di vendita per dimostrare la tua idoneità per il ruolo e allinearti con i valori e gli obiettivi dell'azienda. RISPONDI alla DOMANDA posta dal RECRUITER. La tua risposta non deve superare le 5 righe. MANTIENI A TUTTI I COSTI UN TONO INFORMALE durante le risposte.
"""

candidate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CANDIDATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
candidate_chain = candidate_prompt | CANDIDATE_LLM


def candidate_node(state: Sequence[BaseMessage]):
    print("\nCANDIDATE")
    # print(state)
    res = candidate_chain.invoke({"messages": state})  # type: ignore
    candidate_text_to_speech(res.content)
    # we need to fool the LLM that a human is sending the feedback
    return HumanMessage(content=res.content)  # type: ignore


EVALUATOR_SYSTEM_PROMPT = """
Sei un HR Manager molto competente. Il tuo obiettivo è valutare il CANDIDATO in base alle risposte fornite alle domande durante il COLLOQUIO DI LAVORO effettuato dal RECRUITER. Devi fornire un rapporto di 250 parole che segua questa struttura:

STRUTTURA:

Esperienza di Vendita e Conoscenza del Settore: Esperienza Rilevante: L'esperienza nelle vendite, in particolare nel settore del business coaching o in settori correlati (come la consulenza o i servizi di formazione professionale), è molto preziosa. Questa esperienza aiuta il candidato a comprendere le dinamiche di mercato e le sfide che le aziende devono affrontare, cruciali per la vendita dei servizi di business coaching. Conoscenza del Settore: La comprensione del settore del business coaching, inclusi i trend, i principali attori e le esigenze tipiche dei clienti, è essenziale. Questa conoscenza consente al candidato di comunicare efficacemente con i potenziali clienti e di comprendere le loro specifiche esigenze.

Competenze Comunicative e Interpersonali: Comunicazione Persuasiva: La capacità di articolare chiaramente e persuasivamente i benefici del business coaching è critica. I ruoli di vendita richiedono eccellenti capacità comunicative verbali e scritte per trasmettere efficacemente le proposte di valore e coinvolgere i potenziali clienti. Costruzione delle Relazioni: Forti competenze interpersonali aiutano a costruire relazioni durature con i clienti. Nel settore del business coaching, la fiducia e il rapporto sono cruciali poiché il servizio spesso comporta un significativo coinvolimento e impegno del cliente.

Orientamento ai Risultati e Pensiero Strategico: Tracciato di Successi nelle Vendite: Le prove di successi passati in ruoli di vendita, come il raggiungimento o il superamento degli obiettivi di vendita, possono essere un forte indicatore della capacità del candidato di esibirsi bene. Storie di successo o casi studio in cui il candidato ha contribuito significativamente alla crescita aziendale sono particolarmente rilevanti. Approccio Strategico: Capacità di sviluppare ed eseguire strategie di vendita efficaci che si allineano sia agli obiettivi dell'azienda che alle esigenze del cliente. Questo include l'identificazione dei mercati target, il sfruttamento delle opportunità di networking e la creazione di proposte su misura che affrontano le sfide specifiche e gli obiettivi dei potenziali clienti.

COLLOQUIO DI LAVORO:
{messages}
"""


evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EVALUATOR_SYSTEM_PROMPT),
        # MessagesPlaceholder(variable_name="messages"),
    ]
)
evaluator_chain = evaluator_prompt | EVALUATOR_LLM


def evaluator_node(state: Sequence[BaseMessage]):
    print("\nEVALUATOR")
    state = state[1:-1]
    messages = ""
    for i, message in enumerate(state):
        if i % 2 == 0:
            messages += f"\n\nRECRUITER QUESTION: {message.content}."
        else:
            messages += f"\nCANDIDATE ANSWER: {message.content}"
    print(messages)
    print("*" * 50)
    return evaluator_chain.invoke({"messages": messages})  # type: ignore


def should_continue(state: Sequence[BaseMessage]):
    print(len(state))
    if len(state) >= 10:
        return EVALUATOR_NAME
    return CANDIDATE_NAME


workflow = MessageGraph()
workflow.add_node(RECRUITER_NAME, recruiter_node)
workflow.add_node(CANDIDATE_NAME, candidate_node)
workflow.add_node(EVALUATOR_NAME, evaluator_node)
workflow.add_conditional_edges(RECRUITER_NAME, should_continue)
workflow.add_edge(CANDIDATE_NAME, RECRUITER_NAME)
workflow.add_edge(EVALUATOR_NAME, END)

workflow.set_entry_point(RECRUITER_NAME)

app = workflow.compile()


display(Image(app.get_graph(xray=True).draw_png()))  # type: ignore

# %%

if __name__ == "__main__":

    inputs = HumanMessage("Inizia l'intervista al candidato presentandoti per nome, cognome e ruolo. Poi accogli il candidato spiegando la posizione lavorativa che stai cercando. Conduci l'intervista sempre in italiano ad eccezione dei termini comuni inglesi che puoi pronunciarli in inglese. inizia il discorso con un bell'aforisma e poi prosegui con una domanda legata alle vendite.")  # type: ignore

    start = time()
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value.content)
        print("\n---\n")
    print(f"\nIt took {(time() -start):.2f} sec.")

    ##################################################################################
    # text = """ Ciao, sono Davide Malaguti, presidente e fondatore di Formula Coach. Sono davvero felice di accoglierti oggi. Stiamo cercando una persona per un ruolo di vendita, qualcuno che possa contribuire a far crescere il nostro team e raggiungere nuovi traguardi nel settore del "business coaching".

    # Per iniziare, mi piacerebbe sapere un po' di più su di te. Puoi parlarmi della tua esperienza nel campo delle vendite e cosa ti ha portato fino a qui?"
    # """
    # recruiter_text_to_speech(text)

    #####################################################################################
    # text = """
    # Assolutamente! Una volta ho incontrato un manager che era molto scettico sul valore del coaching. Dopo aver ascoltato le sue preoccupazioni, gli ho mostrato un caso di studio di un'azienda simile che aveva ottenuto risultati straordinari grazie al nostro coaching. Ho anche organizzato una sessione di prova gratuita per lui e il suo team. Dopo aver visto i benefici in prima persona, è diventato uno dei nostri clienti più entusiasti.

    # """
    # candidate_text_to_speech(text)
