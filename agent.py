import os
import time
from dotenv import load_dotenv
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.callbacks import LlamaDebugHandler,CallbackManager
import subprocess


from tools.llm import MYLLM

load_dotenv()


debug_mode = True
llama_debug = LlamaDebugHandler(print_trace_on_end=debug_mode)
callback_manager = None

if debug_mode:
    callback_manager = CallbackManager(handlers=[llama_debug])


#time.sleep(3)
llm = None
#llm_model = "LOCAL_LAMA2CPP"


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

def open_application(application_name:str)->str:
    """
    Opens an application on mac computer with a given text.
    """
    try:
        subprocess.run(["open","-n","-a", application_name])
        return f"Opening {application_name}"
    except Exception as e:
        print(f"Error opening application: {e}")

def open_url(url:str)->str:
    """
    Opens a default browser on mac computer with a given text url.
    """
    try:
        #subprocess.run(["firefox", "--url",url])
        subprocess.Popen(["open", "--url", url])
        return f"Opening {url}"
    except Exception as e:
        print(f"Error opening url: {e}")


if __name__ == "__main__":
    print("**** Hello Agents with Llamaindex ****")
    # llm = MYLLM.get_llm_model()
    llm = MYLLM.get_llm_model("HUGGINGFACE", "default", 0.7)

    tool1 = FunctionTool.from_defaults(fn=write_haiku, name="Write Haiku")
    tool2 = FunctionTool.from_defaults(fn=count_characters, name="Count Characters")
    tool3 = FunctionTool.from_defaults(fn=open_application, name="Open Application")
    tool4 = FunctionTool.from_defaults(fn=open_url, name="Open URL")
    tools_list = [tool1, tool2, tool3, tool4]

    agent = ReActAgent.from_tools(tools=tools_list, llm=llm,verbose=True, callback_manager=callback_manager)
    res = agent.query("Write me a haiku about fennec and then count the characters in it")
    # res = agent.query("Open Obsidian ain my computer")
    # res = agent.query("Open the URL https://www.youtube.com/watch?v=cWc7vYjgnTs in my firefox browser")



    print(res)
    pass