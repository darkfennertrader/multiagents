from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from schemas import AnswerQuestion, ReviseAnswer
from config import set_environment_variables

set_environment_variables()


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are an expert researcher.
        Current time: {time}
        
        1. {first_instruction}
        2. Reflect and Critique your answer. Be severe to maximize improvements.
        3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(time=lambda: datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed report of roughly 250 words."
)
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instruction = """
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


revisor = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, verbose=True)
llm2 = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm2

if __name__ == "__main__":

    ####################     REFLECTION     ###########################

    tweet = ""
    user_query = HumanMessage(content="FIFA World Cup 26")

    print("\nGENERATE")
    for chunk in generate_chain.stream({"messages": [user_query]}):
        print(chunk.content, end="")
        tweet += chunk.content  # type: ignore

    print("\n\nREFLECT")
    reflection = ""
    for chunk in reflect_chain.stream(
        {
            "messages": [
                user_query,
                HumanMessage(content=tweet),
            ]
        }
    ):
        print(chunk.content, end="")
        reflection += chunk.content  # type: ignore

    print("\n\nREGENERATE")
    for chunk in generate_chain.stream(
        {
            "messages": [
                user_query,
                AIMessage(content=tweet),
                HumanMessage(content=reflection),
            ]
        }
    ):
        print(chunk.content, end="")
        tweet += chunk.content  # type: ignore

    ###################     REFLEXION      ###############################

    # human_message = HumanMessage(content="Write about Reinforcement Learning.")

    # # Visualize the parsed tool directly as specified in the class instead of inside the 'tool_calls'
    # chain = (
    #     first_responder_prompt_template
    #     | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    #     | PydanticToolsParser(tools=[AnswerQuestion])
    # )

    # res = chain.invoke(input={"messages": [human_message]})
    # print(res)
