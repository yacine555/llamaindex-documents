import os
import json
import logging
import sys
from dotenv import load_dotenv, find_dotenv

from llama_index.llms import (
    LLM,
    OpenAI,
    LlamaCPP,
    HuggingFaceLLM,
    Replicate
    )
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llama_index.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["REPLICATE_API_TOKEN"] = "REPLICATE_API_TOKEN"

class MYLLM:

    # inject custom system prompt into llama-2

    def custom_completion_to_prompt(completion: str) -> str:
        return completion_to_prompt(
            completion,
            system_prompt=(
                "You are a Q&A assistant. Your goal is to answer questions as "
                "accurately as possible is the instructions and context provided."
            ),
        )

    @staticmethod
    def get_llm_model(llm_model_type:str="OPENAI", llm_model_name:str="text-davinci-003", temperature:float=0.7) -> LLM:
        """
        Select the LLM model to use.

        :param:
            llm_model_type: OPENAI|LOCAL_LAMA2CPP|REPLICATE
            llm_model_name: 
                    for OPENAI: text-davinci-003|gpt-3.5-turbo|gpt-4-1106-preview|gpt-3.5-turbo-1106
            temperature:    
        """

        llm = None

        match llm_model_type:
            case "OPENAI":
                llm = OpenAI(model_name=llm_model_name, temperature=temperature)
            case "LOCAL_LAMA2CPP":
                llm = LlamaCPP(
                        model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf",

                        # optionally, you can set the path to a pre-downloaded model instead of model_url
                        model_path=None,

                        temperature=temperature,
                        max_new_tokens=1024,

                        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                        context_window=2048,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.

                        # kwargs to pass to __call__()
                        generate_kwargs={},

                        # kwargs to pass to __init__()
                        # set to at least 1 to use GPU
                        model_kwargs={"n_gpu_layers": 1},

                        # transform inputs into Llama2 format
                        messages_to_prompt=messages_to_prompt,
                        completion_to_prompt=completion_to_prompt,
                        verbose=True,
                )
            case "REPLICATE":
                # The replicate endpoint
                LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

                llm = Replicate(
                    model=LLAMA_13B_V2_CHAT,
                    temperature=temperature,
                    # override max tokens since it's interpreted
                    # as context window instead of max tokens
                    context_window=4096,
                    # override completion representation for llama 2
                    completion_to_prompt=MYLLM.custom_completion_to_prompt,
                    # if using llama 2 for data agents, also override the message representation
                    messages_to_prompt=messages_to_prompt,
                )

            case "HUGGINGFACE":

                # TODO: only configured to use the HF StableLM LLM model 
                system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
                - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
                - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
                - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
                - StableLM will refuse to participate in anything that could harm a human.
                """

                # This will wrap the default prompts that are internal to llama-index
                query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

                llm = HuggingFaceLLM(
                    context_window=4096,
                    max_new_tokens=256,
                    generate_kwargs={"temperature": temperature, "do_sample": False},
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
                    model_name="StabilityAI/stablelm-tuned-alpha-3b",
                    device_map="auto",
                    stopping_ids=[50278, 50279, 50277, 1, 0],
                    tokenizer_kwargs={"max_length": 4096},
                    # uncomment this if using CUDA to reduce memory usage
                    # model_kwargs={"torch_dtype": torch.float16}
                )

            case _:
                llm = OpenAI(model_name=llm_model_name, temperature=temperature)


        return llm
    

