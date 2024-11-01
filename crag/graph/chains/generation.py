from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

# print(prompt.messages[0])
# print(type(prompt))

generation_chain = prompt | llm | StrOutputParser()
