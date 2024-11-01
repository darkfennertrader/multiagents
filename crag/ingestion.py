import os
from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
)
from langchain_community.vectorstores.chroma import Chroma


from config import set_environment_variables


def list_files_in_directory(directory: str) -> List[str]:
    """
    List all files in the given directory.

    Args:
    directory (str): The path to the directory from which to list files.

    Returns:
    list: A list of file names (with paths) in the specified directory.
    """
    # Check if the specified directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return []

    # List all files in the directory
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))
    ]
    return files


set_environment_variables("MultiAgent_FormulaCoach")


DATA_DIRECTORY = str(Path(__file__).parent / "data")
embeddings = OpenAIEmbeddings()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
# docs = loader.load_and_split(text_splitter=text_splitter)
# print(docs)

# db = Chroma.from_documents(
#     documents=docs,
#     collection_name="rag-malaguti",
#     embedding=embeddings,
#     persist_directory="crag/chroma_db",
# )

retriever = Chroma(
    collection_name="rag-malaguti",
    persist_directory="crag/chroma_db",
    embedding_function=embeddings,
).as_retriever()
