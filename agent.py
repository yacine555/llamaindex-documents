import os
from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool


load_dotenv()
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)

def write_haiku(topic: str) -> str:
    """
    Write a haiku about a given topic.
    """
    haiku = llm.complete(f"Writing a haiku about {topic}")
    return haiku


def count_characters(text:str)->int:
    """
    Counts the number of characters in a text.
    """
    backslash_char = "\\"
    print(f"Counting characters in {text}")
    print(f"Number of characters: {len(text)}")
    print(f"Number of words: {len(text.split())}")
    print(f"Number of lines: {len(text.split('{backslash_char}n'))}")
    print(f"Number of sentences: {len(text.split('.'))}")
    print(f"Number of words per sentence: {len(text.split()) / len(text.split('.'))}")
    print(f"Number of characters per word: {len(text) / len(text.split())}")
    print(f"Number of characters per line: {len(text) / len(text.split('{backslash_char}n'))}")
    print(f"Number of characters per sentence: {len(text) / len(text.split('.'))}")
    print(f"Number of characters per word: {len(text) / len(text.split())}")
    return len(text)


if __name__ == "__main__":
    print("**** Hello Agents with Llamaindex ****")
    

    tool1 = FunctionTool.from_defaults(fn=write_haiku, name="Write Haiku")
    tool2 = FunctionTool.from_defaults(fn=count_characters, name="Count Characters")

    agent = ReActAgent.from_tools(tools=[tool1, tool2], llm=llm,verbose=True)
    res = agent.query("Write me a haiku about fennec and then count the characters in it")
    print(res)
    pass