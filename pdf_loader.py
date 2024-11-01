import os
import shutil
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def remove_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and all its contents
        shutil.rmtree(path)
        print(f"The directory {path} has been removed successfully.")
    else:
        print(f"The directory {path} does not exist.")


def process_llm_response(llm_response):
    # print(llm_response)
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


llm = ChatOpenAI(model="gpt-4o", temperature=0.0, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("data/denoising_diffusion_prob_models.pdf")
docs = loader.load()
# # pages = loader.load()
# # print(len(pages))
# # page = pages[77]
# # print(page.page_content[:30])
# # print(page.metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)


persist_directory = "data/db/chroma"
remove_directory(persist_directory)  # to avoid suplicates for simplicity

vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory=persist_directory
)
print(vectorstore._collection.count())

###################################################

# Load vector store
vector_store = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings
)

# make a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# query = "What is a diffusion model?"
# docs = retriever.invoke(query)

# print(retriever.search_type)
# print(docs[0].page_content)

# chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
)

query = "Tell me more about the progressive coding"
llm_response = qa_chain.invoke({"query": query})
print(process_llm_response(llm_response))

##################################################################
print("\nRAG with LCEL:")
# Load vector store
vector_store = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings
)

# make a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | llm | output_parser


print(chain.invoke("Tell me more about the progressive coding"))
