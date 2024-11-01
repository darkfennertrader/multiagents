from time import time
from graph.graph import app
from crag.config import set_environment_variables
from langgraph.prebuilt import ToolNode, tools_condition

set_environment_variables("Self_RAG")

if __name__ == "__main__":

    QUESTION = "Quale e' il limite maggiore legato alla crescita personale?"
    start = time()
    response = app.invoke(input={"question": QUESTION})
    print(response["generation"])
    print(f"\nIt took {(time() -start):.2f} sec.")
