from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import set_environment_variables

set_environment_variables()

spanish_italian_prompt = ChatPromptTemplate.from_template(
    "Please tell me the spanish and italian words for {word} with an example sentence for each."
)

llm = ChatOpenAI()
output_parser = StrOutputParser()

spanish_italian_chain = spanish_italian_prompt | llm | output_parser  # LCEL

if __name__ == "__main__":

    # INVOKE
    result = spanish_italian_chain.invoke({"word": "polar bear"})
    print(result)

    # # STREAM
    # result = spanish_italian_chain.stream({"word": "polar bear"})
    # for chunk in result:
    #     print(chunk, end="", flush=True)
    # print("\n")

    # BATCH
    # result = spanish_italian_chain.batch([{"word": "polar bear"}, {"word": "giraffe"}])
    # print(result)

    # PROPERTIES
    print()
    print("input_schema", spanish_italian_chain.input_schema.schema())
    print()
    print("output_schema", spanish_italian_chain.output_schema.schema())
