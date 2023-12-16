import os
import time
from dotenv import load_dotenv
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
import subprocess

from tools.llm import MYLLM

load_dotenv()


class MYTOOLBOX:
    llm = MYLLM.get_llm_model("LOCAL_LAMA2CPP", "default", 0.7)

    @staticmethod
    def write_haiku(topic: str) -> str:
        """
        Write a haiku about a given topic.
        """
        haiku = llm.complete(f"Writing a haiku about {topic}")
        print(haiku)
        return haiku

    @staticmethod
    def count_characters(text: str) -> int:
        """
        Counts the number of characters in a text.
        """
        backslash_char = "\\"
        print(f"Counting characters in {text}")
        print(f"Number of characters: {len(text)}")
        print(f"Number of words: {len(text.split())}")
        print(f"Number of lines: {len(text.split('{backslash_char}n'))}")
        print(f"Number of sentences: {len(text.split('.'))}")
        print(
            f"Number of words per sentence: {len(text.split()) / len(text.split('.'))}"
        )
        print(f"Number of characters per word: {len(text) / len(text.split())}")
        print(
            f"Number of characters per line: {len(text) / len(text.split('{backslash_char}n'))}"
        )
        print(f"Number of characters per sentence: {len(text) / len(text.split('.'))}")
        print(f"Number of characters per word: {len(text) / len(text.split())}")
        return len(text)

    @staticmethod
    def open_application(application_name: str) -> str:
        """
        Opens an application on mac computer with a given text.
        """
        try:
            subprocess.run(["open", "-n", "-a", application_name])
            return f"Opening {application_name}"
        except Exception as e:
            print(f"Error opening application: {e}")

    @staticmethod
    def open_url(url: str) -> str:
        """
        Opens a default browser on mac computer with a given text url.
        """
        try:
            # subprocess.run(["firefox", "--url",url])
            subprocess.Popen(["open", "--url", url])
            return f"Opening {url}"
        except Exception as e:
            print(f"Error opening url: {e}")
