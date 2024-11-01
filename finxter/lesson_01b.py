from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import set_environment_variables

set_environment_variables("Test LangChain")

spanish_italian_prompt = ChatPromptTemplate.from_template(
    "Please tell me the spanish and italian words for {word} with an example sentence for each."
)

check_if_correct_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that looks at a question and it's given answer. You will find out what is wrong with the answer and improve it. You will return the improved version of the answer.
    Question:\n{question}\nAnser Given:\n{initial_answer}\nReview the answer give me an improved version instead.
    Improved answer:
    """
)

llm = ChatOpenAI()
output_parser = StrOutputParser()

spanish_italian_chain = spanish_italian_prompt | llm | output_parser  # LCEL
check_answer_chain = check_if_correct_prompt | llm | output_parser


def run_chain(word: str) -> str:
    initial_answer = spanish_italian_chain.invoke({"word": word})
    print(f"initial answer: {initial_answer}", end="\n\n")

    answer = check_answer_chain.invoke(
        {
            "question": f"Please tell me the spanish and italian words for {word} with an example sentence for each.",
            "initial_answer": initial_answer,
        }
    )

    print(f"\nimproved answer: {answer}")

    return answer


if __name__ == "__main__":
    run_chain("strawberries")
