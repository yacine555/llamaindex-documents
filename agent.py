import os
import time
from dotenv import load_dotenv
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.callbacks import LlamaDebugHandler,CallbackManager
import subprocess

from tools.llm import MYLLM
from tools.agent import MYAGENT

load_dotenv()


debug_mode = True
llama_debug = LlamaDebugHandler(print_trace_on_end=debug_mode)
callback_manager = None

if debug_mode:
    callback_manager = CallbackManager(handlers=[llama_debug])


#time.sleep(3)
llm = None
#llm_model = "LOCAL_LAMA2CPP"


if __name__ == "__main__":
    print("**** Hello Agents with Llamaindex ****")
    # llm = MYLLM.get_llm_model()
    llm = MYLLM.get_llm_model("LOCAL_LAMA2CPP", "default", 0.7)

    tool1 = FunctionTool.from_defaults(fn=MYAGENT.write_haiku, name="Write Haiku")
    tool2 = FunctionTool.from_defaults(fn=MYAGENT.count_characters, name="Count Characters")
    tool3 = FunctionTool.from_defaults(fn=MYAGENT.open_application, name="Open Application")
    tool4 = FunctionTool.from_defaults(fn=MYAGENT.open_url, name="Open URL")
    tools_list = [tool1, tool2, tool3, tool4]

    agent = ReActAgent.from_tools(tools=tools_list, llm=llm,verbose=True, callback_manager=callback_manager)
    res = agent.query("Write me a haiku about fennec and then count the characters in it")
    # res = agent.query("Open Obsidian ain my computer")
    # res = agent.query("Open the URL https://www.youtube.com/watch?v=cWc7vYjgnTs in my firefox browser")



    print(res)


    response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
    print(response.text)

    pass