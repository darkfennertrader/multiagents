# %%
from time import time, sleep
import asyncio
from typing import Annotated, TypedDict, List, Sequence
from IPython.display import Image, display
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from elevenlabs import play, stream, save, Voice, VoiceSettings
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)
from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from langgraph.graph import END, MessageGraph, StateGraph

from config import set_environment_variables

set_environment_variables("Candidate_Interview")


client = ElevenLabs()

chunks = ""


def check_elements(list1, list2=[",", ", ", ".", ". "]):
    for element in list1:
        if element in list2:
            return True
    return False


def stream_text(text, delay=0.03):
    for char in text:
        print(char, end="", flush=True)
        sleep(delay)


def streaming_mode(chain, state):
    global chunks
    chunks = ""
    chunks_to_yield = []
    for chunk in chain.stream({"messages": state}):
        print(chunk.content, end="", flush=True)
        chunks += chunk.content
        chunks_to_yield.append(str(chunk.content))

        # Check if we have accumulated 3 chunks
        # print(chunks_to_yield)
        # print(check_elements(chunks_to_yield))
        if len(chunks_to_yield) >= 3 and check_elements(chunks_to_yield):
            # print(chunks_to_yield)
            # print(check_elements(chunks_to_yield))
            yield "".join(chunks_to_yield)
            chunks_to_yield = []  # Reset the list after yielding
            sleep(1)

    # If there are remaining chunks that haven't been yielded because they didn't make up a full group of 3
    if chunks_to_yield:
        yield "".join(chunks_to_yield)


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
        optimize_streaming_latency=0,
        voice=Voice(
            voice_id="d2rfZk3QRo5vHPDOBYlA",
            settings=VoiceSettings(
                stability=1, similarity_boost=1, use_speaker_boost=True, style=0.0
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
        optimize_streaming_latency=0,
        voice=Voice(
            voice_id="CmiOZ78hHwigIq11xTcn",
            settings=VoiceSettings(
                stability=0.8, similarity_boost=1, use_speaker_boost=True, style=0.0
            ),
        ),
    )
    return stream(audio_stream)


def evaluator_text_to_speech(text):
    """
    Use this tools to transform text to speech
    """
    # text = state["message"][-1].content
    audio_stream = client.generate(
        text=text,
        model="eleven_multilingual_v2",  # "eleven_multilingual_v2"
        stream=True,
        voice=Voice(
            voice_id="kmIocz8ptnzGYxNhfW6f",
            settings=VoiceSettings(
                stability=0.8, similarity_boost=1.0, use_speaker_boost=False, style=0.0
            ),
        ),
    )
    return stream(audio_stream)


RECRUITER_NAME = "interviewer"
CANDIDATE_NAME = "candidate"
EVALUATOR_NAME = "evaluator"
POSITION = "District Manager"
FUNCTION = "Vendite"

MEMBERS = [RECRUITER_NAME, CANDIDATE_NAME]

RECRUITER_LLM = ChatOpenAI(model="gpt-4o", temperature=0.6, streaming=True)
CANDIDATE_LLM = ChatOpenAI(model="gpt-4o", temperature=0.4, streaming=True)
EVALUATOR_LLM = ChatOpenAI(model="gpt-4-turbo", temperature=0.3, streaming=True)
TAVILY_TOOL = TavilySearchResults(max_results=3)


# You have access to a web search function for additional or up-to-date research if needed. Use this tool to make questions that are relevant to your industry and make use of the the most up-to-date techniques in the field of human resources.
# """


RECRUITER_SYSTEM_PROMPT = """
Sei Davide Malaguti il presidente e fondatore di Formula Coach con oltre 30 anni di esperienza alle spalle nella consulenza strategiga nel campo delle risorse umane alle imprese. Formula Coach e' un'azienda che opera nel settore del business coaching. Sei incaricato di valutare un candidato per un ruolo di {role} in ambito {function}. Il tuo obiettivo è determinare se il candidato Lorenzo Neri possiede le competenze, l'esperienza e l'attitudine necessarie per eccellere in questa posizione. Per raggiungere questo scopo, coinvolgerai il candidato in un'intervista strutturata, concentrandoti sulla sua esperienza nel/nella {function}, sulla comprensione dell'industria del business coaching e sulla capacità di raggiungere gli obiettivi legati alla {function}. Valuterai anche le sue capacità comunicative, di risoluzione dei problemi e il suo approccio alla costruzione e al mantenimento delle relazioni con i clienti. Le tue domande dovrebbero essere progettate per ottenere risposte dettagliate che forniscano una visione delle capacità del candidato e della sua idoneità per il ruolo. Rispondi alla risposta fornita dal CANDIDATO se presente, altrimenti fai una domanda SFIDANTE!. Quando fornisci la domanda devi essere conciso (massimo tre righe) e DEVI ESSERE MOLTO AGGRESSIVO E METTERE SEMPRE IN DIFFICOLTA' IL CANDIDATO CHE INTERVISTI. "Pensa bene se il candidato ha risposto o meno alla tua domanda ed in caso non lo abbia fatto metti in evidenza questa cosa.". Le domande devono essere sempre in lingua italiana. 
"""

recruiter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RECRUITER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
recruiter_prompt_refined = recruiter_prompt.partial(role=POSITION, function=FUNCTION)

recruiter_chain = recruiter_prompt_refined | RECRUITER_LLM


async def recruiter_node(state: Sequence[BaseMessage]):
    print("\nRECRUITER")
    print("*" * 10)
    # print(state)
    if len(state) >= 9:
        text = "  Grazie per il tuo tempo e la tua disponibilita'. Nelle prossime settimane termineremo il processo di selezione e ti faremo sapere. In bocca al lupo!"
        stream_text(text)
        res = recruiter_text_to_speech(text)
        # print(res)
        return AIMessage(content=text)

    # response = recruiter_chain.invoke({"messages": state})  # type: ignore
    # print(response)
    # print(type(response))
    # recruiter_text_to_speech(response.content)
    recruiter_text_to_speech(streaming_mode(recruiter_chain, state))
    # print(chunks)
    # return response
    return AIMessage(content=chunks)


CANDIDATE_SYSTEM_PROMPT = """
Ti chiami Lorenzo Neri e ti stai candidando per un ruolo di {role} nella funzione {function} nel settore della consulenza stategica delle Risorse Umane. Il tuo obiettivo è distinguerti dagli altri candidati fornendo risposte convincenti e perspicaci alle domande poste dal RECRUITER. Sei una persona consapevole dei propri limiti ma anche delle grandi potenzialita'. Hai ottime capacita' comunicative, relazionalie  di gestione delle obiezioni. Sei un ex avvocato illuminato dalla passione per i rapporti umani e desideroso di scoprire tutti i segreti della consulenza nelle risorse umane. Sfrutta la tua esperienza precedente, la comprensione del settore e la tua filosofia personale per dimostrare la tua idoneità per il ruolo e allinearti con i valori e gli obiettivi dell'azienda. NON inventare situazioni se non hai una esperienza di lavoro diretta nella funzione ricercata. RISPONDI sempre alla DOMANDA posta dal RECRUITER. La tua risposta non deve superare le 3 righe e deve essere sempre in lingua italiana. Saluta il recruiter solo all'inizio della conversazione. Non salutarlo dopo ogni risposta ma solo la prima volta. Ricordati di essere sempre il CANDIDATO. Non fare mai domande!
"""

candidate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CANDIDATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
candidate_prompt_refined = candidate_prompt.partial(role=POSITION, function=FUNCTION)
candidate_chain = candidate_prompt_refined | CANDIDATE_LLM


async def candidate_node(state: Sequence[BaseMessage]):
    print("\nCANDIDATE")
    print("*" * 10)
    # print(state)
    # res = candidate_chain.invoke({"messages": state})  # type: ignore
    # candidate_text_to_speech(res.content)
    candidate_text_to_speech(streaming_mode(candidate_chain, state))
    # we need to fool the LLM that a human is sending the feedback
    return HumanMessage(content=chunks)  # type: ignore


EVALUATOR_SYSTEM_PROMPT = """
Sei un HR Manager molto competente. Il tuo obiettivo è valutare il CANDIDATO in base alle risposte fornite alle domande durante il COLLOQUIO DI LAVORO effettuato dal RECRUITER. Devi fornire un rapporto di massimo 50 parole in Italiano che segua questa struttura:

STRUTTURA:

\n\nEsperienza Rilevante e Conoscenza del Settore:
\nEsperienza Rilevante: 
\nL'esperienza nel settore del business coaching o in settori correlati (come la consulenza o i servizi di formazione professionale), è molto preziosa. Questa esperienza aiuta il candidato a comprendere le dinamiche di mercato e le sfide che le aziende devono affrontare, cruciali per la vendita dei servizi di business coaching.
\nConoscenza del Settore: 
\nLa comprensione del settore del business coaching, inclusi i trend, i principali attori e le esigenze tipiche dei clienti, è essenziale. Questa conoscenza consente al candidato di comunicare efficacemente con i potenziali clienti e di comprendere le loro specifiche esigenze.

\n\nCompetenze Comunicative e Interpersonali:
\nComunicazione Persuasiva: 
\nLa capacità di articolare chiaramente e persuasivamente i benefici del business coaching è critica. I ruoli di vendita richiedono eccellenti capacità comunicative verbali e scritte per trasmettere efficacemente le proposte di valore e coinvolgere i potenziali clienti. 
\nCostruzione delle Relazioni:
\nForti competenze interpersonali aiutano a costruire relazioni durature con i clienti. Nel settore del business coaching, la fiducia e il rapporto sono cruciali poiché il servizio spesso comporta un significativo coinvolimento e impegno del cliente.

\n\nOrientamento ai Risultati e Pensiero Strategico:
\nTracciato di Successi:
\nLe prove di successi passati in ruoli di vendita, come il raggiungimento o il superamento degli obiettivi di vendita, possono essere un forte indicatore della capacità del candidato di esibirsi bene. Storie di successo o casi studio in cui il candidato ha contribuito significativamente alla crescita aziendale sono particolarmente rilevanti.
\nApproccio Strategico: 
\nCapacità di sviluppare ed eseguire strategie di vendita efficaci che si allineano sia agli obiettivi dell'azienda che alle esigenze del cliente. Questo include l'identificazione dei mercati target, il sfruttamento delle opportunità di networking e la creazione di proposte su misura che affrontano le sfide specifiche e gli obiettivi dei potenziali clienti.

\n\nCONCLUSIONI ED ESITO:
Sintetizza le informazioni precedenti e dai un esito finale
\nESITO:
\nPOSITIVO o NEGATIVO (non entrambi)


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


async def evaluator_node(state: Sequence[BaseMessage]):
    print("\nEVALUATOR")
    print("*" * 10)
    state = state[1:-1]
    messages = ""
    for i, message in enumerate(state):
        if i % 2 == 0:
            messages += f"\n\nRECRUITER QUESTION: {message.content}."
        else:
            messages += f"\nCANDIDATE ANSWER: {message.content}"
    # print(messages)

    response = evaluator_chain.invoke({"messages": messages})  # type: ignore
    # print(response.content)
    evaluator_text_to_speech(streaming_mode(evaluator_chain, state))
    # evaluator_text_to_speech(response.content)
    return response


def should_continue(state: Sequence[BaseMessage]):
    # print(len(state))
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


async def main(inputs):
    start = time()
    async for output in app.astream(input=inputs, stream_mode="updates"):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            # print("---")
            # print(value.content)
        print("\n----------\n")

    print(f"\nIt took {(time() -start):.2f} sec.")


# %%


if __name__ == "__main__":

    inputs = HumanMessage(
        content="""Inizia l'intervista al candidato presentandoti per nome, cognome e ruolo. Poi accogli il candidato Lorenzo Neri ringraziandolo e spiegando la posizione lavorativa che stai cercando.
        Segui le seguenti indicazioni:
        Conduci l'intervista sempre in italiano. Inizia il discorso con un bell'aforisma e prosegui con una domanda legata al ruolo."""
    )

    asyncio.run(main(inputs))

    ##################################################################################
    # text = """ Ciao, sono Davide Malaguti, presidente e fondatore di Formula Coach. Sono davvero felice di accoglierti oggi. Stiamo cercando una persona per un ruolo di vendita, qualcuno che possa contribuire a far crescere il nostro team e raggiungere nuovi traguardi nel settore del "business coaching".

    # Per iniziare, mi piacerebbe sapere un po' di più su di te. Puoi parlarmi della tua esperienza nel campo delle vendite e cosa ti ha portato fino a qui?"
    # """
    # LLM = ChatOpenAI(model="gpt-4o", temperature=0.6, streaming=True)
    # prompt = ChatPromptTemplate.from_template("{context}")
    # parser = StrOutputParser()
    # chain = prompt | LLM

    # def streaming_mode():
    #     for chunk in chain.stream({"context": text}):
    #         yield chunk.content

    # recruiter_text_to_speech(streaming_mode())

    # recruiter_text_to_speech(resp)

    #####################################################################################
    # text = """
    # Assolutamente! Una volta ho incontrato un manager che era molto scettico sul valore del coaching. Dopo aver ascoltato le sue preoccupazioni, gli ho mostrato un caso di studio di un'azienda simile che aveva ottenuto risultati straordinari grazie al nostro coaching. Ho anche organizzato una sessione di prova gratuita per lui e il suo team. Dopo aver visto i benefici in prima persona, è diventato uno dei nostri clienti più entusiasti.

    # """
    # candidate_text_to_speech(text)
