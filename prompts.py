import langchain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

print("\nLangChain Version: ", langchain.__version__, "\n")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0, verbose=False)

# Create a new prompt template using initializer
template = PromptTemplate(
    template="Hello, my name is {name}. I am a {profession}.",
    input_variables=["name", "profession"],
)

# Use the template to generate a prompt
prompt = template.format(name="Alice", profession="software developer")
print(prompt)


# Create a new template using instantiation
prompt_template = PromptTemplate.from_template(
    template="Hello, my name is {name}. I am a {profession}."
)

# Use the template to generate a prompt
output = template.format(name="Bob", profession="software developer")
print(output)

print("\nOutput from chain:")
chain = prompt_template | llm
res = chain.invoke(input={"name": "Bob", "profession": "software developer"})
print(res)
