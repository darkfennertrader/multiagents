# pylint: disable=no-self-argument

from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer the question. The output of this class MUST be in the following format:
    {
        "answer": "..."
        "reflection": {
            "missing": "...",
            "superfluous": "..."
        }
        "search_queries": [...]
    }
    """

    answer: str = Field(description="~500 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 internet search queries for researching improvements to address the critique of your current answer."
    )

    # # adding validation logic
    # @validator("search_queries")
    # def check_num_queries(cls, field):
    #     if int(len(field)) <= 0:
    #         raise ValueError("No search queries generated.")
    #     return field


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
