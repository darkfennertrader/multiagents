from typing import Dict, Any
from crag.graph.state import GraphState
from crag.graph.chains.retrieval_grader import retrieval_grader, GradeDocument


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determnins whther the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (GraphState): The current graph state

    Returns:
        Dict[str, Any]: Filtered out irrelevant documents and updated web_search state
    """
    print("\nCHECKING DOCUMENTS RELEVANCE TO QUESTION")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        score: GradeDocument = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}  # type: ignore
        )
        if score.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
