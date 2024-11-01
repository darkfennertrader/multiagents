from typing import Dict, Any
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from crag.graph.state import GraphState


web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("\nWEB SEARCH")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    # print(tavily_results)
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    # print(joined_tavily_results)
    web_results = Document(page_content=joined_tavily_results)
    print(web_results)

    if documents:
        documents.append(str(web_results))
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "gelato", "documents": None})  # type: ignore
