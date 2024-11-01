from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        return print(f"\nPROMPT to LLM was: \n{messages[0][0].content} \n ********")

    # def on_llm_end(
    #     self,
    #     response: LLMResult,
    #     **kwargs: Any,
    # ) -> Any:
    #     return print(f"\nLLM response was: \n{response.generations[0][0].text}")
