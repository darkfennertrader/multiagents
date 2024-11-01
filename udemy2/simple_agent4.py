from typing import Any
import re
from openai import OpenAI
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


# Function to handle the interactive query
def query_interactive():
    bot = SimpleAgent(prompt)
    max_turns = int(input("Enter the maximum number of turns: "))
    i = 0

    while i < max_turns:
        i += 1
        question = input("You: ")
        result = bot(question)
        print("Bot:", result)

        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]  # type: ignore
        if actions:
            action, action_input = actions[0].groups()  # type: ignore
            if action not in known_actions:
                print(f"Unknown action: {action}: {action_input}")
                continue
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
            result = bot(next_prompt)
            print("Bot:", result)
        else:
            print("No actions to run.")
            break


if __name__ == "__main__":
    query_interactive()
