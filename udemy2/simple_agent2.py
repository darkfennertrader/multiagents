from typing import Any
from openai import OpenAI
from udemy2.prompts import prompt, planet_mass, calculate
from udemy2.config import set_environment_variables

set_environment_variables("Simple_Agent")


client = OpenAI()


class SimpleAgent:
    def __init__(self, system="") -> None:
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.chat.completions.create(
            model="gpt-4o", temperature=0.0, messages=self.messages
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    question = "What is the combined mass of Earth and Jupiter?"
    print(f"Question: {question}")
    agent = SimpleAgent(system=prompt)
    response = agent(message=question)
    print(response)
    response = planet_mass("Earth")
    print(response)
    next_response = f"Observation: {response}"
    print(next_response)
    response = agent(message=next_response)
    print(response)
    response = planet_mass("Jupiter")
    print(response)
    response = f"Observation: {response}"
    response = agent(message=response)
    print(response)
    response = f"Observation: {eval('5.972 + 1898.19')}"
    print(response)

    response = agent(message=response)
    print(response)

    # print()
    # print(agent.messages)
