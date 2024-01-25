import os
from dotenv import load_dotenv

from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers import SimpleWebPageReader
from llama_index import (
    SimpleDirectoryReader,
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
import pinecone
import torch
import nltk
import ssl


"""
ssl verification for ntkl resources download localy
"""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


"""
Dowload the Natural Toolkit resources on first use if not available
"""
# nltk.download()

load_dotenv()

index_name = os.environ["PINECONE_INDEX_NAME"]
embeddings = os.environ["EMBEDDING_CONFIG"]  # DEFAULT => OPENAI | LOCAL => Huggingface

llm = None
llm_model_type = "DEFAULT"  # DEFAULT => OPENAI | LOCAL => Huggingface

def scrapURL(url: str) -> VectorStoreIndex:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    print(documents)
    index = VectorStoreIndex.from_documents(documents=documents)
    return index

def load_data() -> VectorStoreIndex:
    """
    Load structured data (md file) using OpenAI llm and default inMemory storage
    """
    reader = SimpleDirectoryReader(input_dir="./docs/streamlit-docs", recursive=True)
    docs = reader.load_data()

    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.",
        )
    )
    vsindex = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return vsindex


def load_unstructured_data():
    """
    Load unstructured data into Pinecone using OpenAI llm
    """
    UnstructuredReader = download_loader("UnstructuredReader")

    dir_reader = SimpleDirectoryReader(
        "./docs/llamindex-docs",
        file_extractor={
            ".pdf": UnstructuredReader(),
            ".html": UnstructuredReader(),
            ".eml": UnstructuredReader(),
        },
    )
    documents = dir_reader.load_data()
    # print(documents[0].text)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    # nodes = node_parser.get_nodes_from_documents(documents=documents)

    service_context = None

    if embeddings == "DEFAULT":
        print("Using OpenAI embeddings...")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002", embed_batch_size=100
        )
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser
        )
    else:
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
            generate_kwargs={"temperature": 0.7, "do_sample": False},
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

        # Using local embeddings downloaded from Huggungface with HGF LLM
        print("Using Local embeddings...")
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-large-en", node_parser=node_parser
        )

    pinecone_index = pinecone.Index(index_name=index_name)
    vectore_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vectore_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )

    print("Finish Ingesting...")


if __name__ == "__main__":
    # vsindex=load_data()
    #load_unstructured_data()
    #scrapURL("https://mistral.ai")

