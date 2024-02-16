import os
from pydantic import BaseModel
from typing import Dict, Iterable, List

import numpy as np

from .chatgpt import AzureChatGPT



class MemoryEvent(BaseModel):
    description: str
    embedding: List[float] = None


class AssistantMemory:
    def __init__(self):
        self.events = []

    def _embed(self, text):
        import openai
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def store(self, event: MemoryEvent):
        embed = self._embed(event.description)
        event = event.model_copy()
        event.embedding = embed
        self.events.append(event)

        print("\n\nALL SAVED MEMORIES:")
        for event in self.events:
            print(event.description)

    def retrieve(self, query, topk) -> List[MemoryEvent]:
        qembed = np.array(self._embed(query)).reshape(-1, 1)
        embeds = np.stack([np.array(event.embedding) for event in self.events])
        sims = (embeds @ qembed).flatten()
        order = np.argsort(sims)[::-1]
        inds = order[:topk]
        return [self.events[ind] for ind in inds]


class StatefulAssistant:

    def __init__(self, name: str, instructions: list[str], model: str = "gpt-4"):
        self.name = name
        self.instructions = instructions
        self.memory = AssistantMemory()
        self.model = model

        self.memory.store(MemoryEvent(description="Test memory"))

    def add_instruction(self, instruction: str):
        self.instructions.append(instruction)

    def chat(self, messages: List[Dict[str, str]], like_api=False):
        gpt = AzureChatGPT(self.model, assistant_name=self.name)

        system_message = f"You are a personal assistant named {self.name}.\n\n"
        system_message += "Please follow the instructions below:\n\n"
        system_message += "- " + "\n- ".join(self.instructions)
        system_message += "\n\nHere are some of your most similar interactions with the user for additional context:\n\n"
        system_message += "\n\n".join([f"```\n{event.description}\n```" for event in self.memory.retrieve(messages[-1]['content'], topk=20)])
        print("SYSTEM MESSAGE:")
        print(system_message)

        gpt.system(system_message)
        gpt._messages = messages

        self.memory.store(MemoryEvent(
            description=messages[-1]['content']
        ))

        if like_api:
            def stream_wrapper():
                stream = gpt._make_completion_stream(like_api=True)
                text_chunks = []
                for chunk in stream:
                    yield chunk

                    if chunk is not None and chunk.choices[0].delta.content is not None:
                        text_chunks.append(chunk.choices[0].delta.content)

                
            return stream_wrapper()
        
        return gpt.call(stream=True)
