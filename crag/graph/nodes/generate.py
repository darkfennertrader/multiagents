from typing import Dict, Any
from crag.graph.state import GraphState
from crag.graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    print("\nGENERATE:")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}
