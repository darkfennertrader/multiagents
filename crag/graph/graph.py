# %%
from langgraph.graph import END, StateGraph
from crag.graph.chains.hallucination_grader import hallucination_grader
from crag.graph.chains.answer_grader import answer_grader
from crag.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from crag.graph.nodes import retrieve, grade_documents, web_search, generate
from crag.graph.state import GraphState


def decide_to_generate(state: GraphState):
    print("\nDECIDE TO GENERATE?")

    if state["web_search"]:
        print("not all docs are relevant to question")
        return WEBSEARCH
    else:
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("\nCHECK ALLUCINATION")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        print("GRADE GENERATION vs QUESTION")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("GENERATION ADDRESS QUESTION")
            return "useful"
        else:
            return "not useful"
    else:
        print("DECISION IS NOT GROUNDED IN DOCUMENTS")
        return "not supported"


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(WEBSEARCH, GENERATE)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"not supported": GENERATE, "not useful": WEBSEARCH, "useful": END},
)

workflow.add_edge(GENERATE, END)

app = workflow.compile()

# Image(app.get_graph().draw_png(output_file_path="crag.png"))  # type: ignore

# %%
