from operator import itemgetter
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o", temperature=0.9, verbose=True)

prompt = PromptTemplate(
    input_variables=["language"], template="How do you say good morning in {language}"
)

chain = prompt | llm

print(chain.invoke({"language": "Italian"}).content)


##############################################################################
print("\nSEQUENTIAL CHAIN:")

template = """
As a children's book writer, please come up with a simple and short (90 words) lullaby based on the location {location} and the main character {name}.
STORY:
"""
prompt1 = PromptTemplate(
    input_variables=["locations", "name"],
    template=template,
)

chain1 = prompt1 | llm

print()
# print(chain1.invoke({"location": "Italy", "name": "princess Chiara"}).content)


template_update = """
Translate the {story} into {language}. Make sure the language is simple and fun.

TRANSLATION:
"""

prompt2 = PromptTemplate(
    input_variables=["story", "language"],
    template=template_update,
)

chain2 = prompt2 | llm

sequential_chain = {"story": chain1, "language": itemgetter("language")} | chain2
print()
print(
    sequential_chain.invoke(
        {"location": "Italy", "name": "princess Chiara", "language": "Italian"}
    ).content
)


##############################################################################
print("\nROUTER CHAIN:")
