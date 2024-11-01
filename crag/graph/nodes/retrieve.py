from typing import Any, Dict
from crag.graph.state import GraphState
from crag.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("\nRETRIEVE NODE:")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
