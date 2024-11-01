from typing import Any
from openai import OpenAI
import re
from udemy2.prompts import prompt, planet_mass, calculate
from udemy2.config import set_environment_variables

set_environment_variables("Simple_Agent")


action_re = re.compile(r"^Action: (\w+): (.*)$")


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


known_actions = {"calculate": calculate, "planet_mass": planet_mass}


# Create a query function
def query(question, max_turns=10):
    i = 0
    bot = SimpleAgent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]  # type: ignore
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()  # type: ignore
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            return


if __name__ == "__main__":
    question = "What is the combined mass of Earth and Jupiter and Saturn?"
    print(f"Question: {question}")
    query(question)

    # print()
    # print(agent.messages)
