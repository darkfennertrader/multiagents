import functools
import operator
from typing import Annotated, Sequence, TypedDict, List
from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph


llm = ChatOpenAI(model="gpt-4o", temperature=0)

email_response = """
Hello, we are a family of 4 people and we would like to visit the following cities: Milan, Rome and Naples, Can you please book a room for us.
many thanks
Raimondo
"""


email_template = """
from the following email, extract the following information:

number_of_people: this is the number of people to accomodate in the hotel.
cities_to_visit: extract the cities they are going to visit. For more than one city put them in square brackets like this: '[city1, city2]'.
email_writer: extract the name of the person who wrote the email.

Format the output as JSON

email: {email}
"""

print()
prompt_template = ChatPromptTemplate.from_template(email_template)

print(prompt_template)

messages = prompt_template.format_messages(email=email_response)
print("\nOutput from the LLM:")
print(llm.invoke(input=messages).content)  # output is a string

number_of_people_schema = ResponseSchema(
    name="number_of_people",
    description="number of people in the email. It's usally a number. If not available write n/a",
)

cities_to_visit_schema = ResponseSchema(
    name="cities_to_visit",
    description="These are the cities that will be visisted. This must be in a list. If not available write n/a",
)

email_writer_schema = ResponseSchema(
    name="email_writer",
    description="It's the name of the person who undersigns the email. If not available write n/a",
)

response_schema = [number_of_people_schema, cities_to_visit_schema, email_writer_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)

format_instruction = output_parser.get_format_instructions()

# print(format_instruction)

email_template_revised = """
from the following email, extract the following information:

number_of_people: this is the number of people to accomodate in the hotel.
cities_to_visit: extract the cities they are going to visit. For more than one city put them in square brackets like this: '[city1, city2]'.
email_writer: extract the name of the person who wrote the email.

Format the output as JSON

email: {email}
{format_instructions}
"""

updated_prompt = ChatPromptTemplate.from_template(email_template_revised)
messages = prompt_template.format_messages(email=email_response)
