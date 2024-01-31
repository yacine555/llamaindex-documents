import os
from dotenv import load_dotenv

from llama_index.llms.base import BaseLLM
from llama_index.llms import (
    HuggingFaceLLM,
    LangChainLLM,
    Ollama,
    OpenAI
)

from llama_index import Prompt
from llama_index.prompts import PromptTemplate


text_qa_template = Prompt(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_template = Prompt(
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)


def get_llm(type:int, model_name:str, temperature=0, verbose=False, system_prompt:str=None, **kwargs) -> BaseLLM:
    """
        Type:   1 - OppenAI
                2 - Local Ollama
                3 - HuggingFace LLM
        model_name: name of the moodel to use

    """
    llm = None

    match type:
        case 1:
            llm = OpenAI(temperature=temperature, model=model_name, verbose = verbose)
        case 2:
            llm = Ollama(model=model_name,temperature = temperature,verbose=verbose)
        case 3:
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
            print("The LLM type was not defined")
    return llm




