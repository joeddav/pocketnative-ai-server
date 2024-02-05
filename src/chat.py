import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .chatgpt import AzureChatGPT



@dataclass
class MemoryEvent:
    assistant_name: str
    metadata: dict
    event_description: str


class AssistantMemory:
    def __init__(self, assistant_name, filters=None):
        self.assistant_name = assistant_name
        self.filters = filters or dict()

    def store(self, event: MemoryEvent):
        pass

    def retrieve(self, k) -> List[MemoryEvent]:
        pass


class StatefulAssistant:

    def __init__(self, name: str, instructions: list[str], model: str = "gpt-4"):
        self.name = name
        self.instructions = instructions
        self.memory = AssistantMemory(name)
        self.model = model

    def add_instruction(self, instruction: str):
        self.instructions.append(instruction)

    def chat(self, messages: List[Dict[str, str]], like_api=False):
        gpt = AzureChatGPT(self.model)

        system_message = f"You are a personal assistant named {self.name}.\n\n"
        system_message += "Please follow the instructions below:\n\n"
        system_message += "-" + "\n- ".join(self.instructions)

        gpt.system(system_message)
        gpt._messages = messages

        if like_api:
            return gpt._make_completion_stream(like_api=True)
        
        return gpt.call(stream=True)
