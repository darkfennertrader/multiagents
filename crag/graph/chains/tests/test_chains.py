import subprocess
from pprint import pprint
from crag.graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from crag.graph.chains.generation import generation_chain
from crag.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)
from crag.ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "gelato"
    docs = retriever.invoke(question)
    # print(docs)
    doc_txt = docs[0].page_content
    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )  # type: ignore
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "Machine Learning"
    docs = retriever.invoke(question)
    # print(docs)
    doc_txt = docs[0].page_content
    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )  # type: ignore
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "gelato"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "gelato"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "Machine Learning"
    docs = retriever.invoke(question)
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "you are completely useless"}
    )
    assert not res.binary_score


# test_generation_chain()

if __name__ == "__main__":
    # execute tests within a python script
    command = "pytest crag/ -s -v"
    try:
        # Run the command with check=True to raise an exception on error
        process = subprocess.run(
            command, shell=True, text=True, capture_output=True, check=True
        )
        print("Command executed successfully!")
        print("Output:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code:", e.returncode)
        print("Error output:")
        print(e.stderr)
