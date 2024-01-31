import os
import os.path
import logging
import sys
from typing import List

from dotenv import load_dotenv

# Make the printing look nice
from llama_index.schema import (
    Document,
    MetadataMode
)

from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import (
    Prompt,
    PromptTemplate
)
from .llm import get_llm

from llama_index.embeddings import (
    HuggingFaceEmbedding,
    MistralAIEmbedding,
    OpenAIEmbedding,   
)

from data.markdown_docs_reader import MarkdownDocsReader
from llama_index.readers import SimpleWebPageReader
from llama_index import (
    download_loader,
    load_index_from_storage,
    Response,
    ServiceContext,
    set_global_service_context,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)

from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores import ChromaVectorStore
import chromadb
import pinecone
import torch
import nltk
import ssl

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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

# Pytorch configuration to only use CPU
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
SOURCE_DIR = "./data/source/"
PERSIST_DIR = "./data/storage/"
index_name = os.environ["PINECONE_INDEX_NAME"]
embeddings = os.environ["EMBEDDING_CONFIG"]  # DEFAULT => OPENAI | LOCAL_HF => Huggingface | LOCAL_OLLAMA

llm = None
llm_model_type = "DEFAULT"  # DEFAULT => OPENAI | LOCAL_HF => Huggingface | LOCAL_OLLAMA


def get_service_context(llm,chunk_size:int = 1000) -> ServiceContext:
    """

    """
    embed_model = None
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    if embeddings == "DEFAULT":

        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002", embed_batch_size=100
        )
    else:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser
    )
 
    return service_context

def get_storage_context() -> StorageContext:
    pass

def scrapURL(url: str) -> VectorStoreIndex:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    print(documents)
    index = VectorStoreIndex.from_documents(documents=documents)
    return index

def load_markdown_docs(filepath) -> List[Document]:
    """Load markdown docs from a directory, excluding all other file types."""

    filepath = SOURCE_DIR + filepath
    loader = SimpleDirectoryReader(
        input_dir= filepath,
        required_exts= [".md"],
        # exclude=["*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg", "*.jpeg", "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive= True
    )

    documents = loader.load_data()

    # customize the formating so that the LLM and embedding models can get a better idea of what they are reading.
    text_template = "Content Metadata:\n{metadata_str}\n\nContent:\n{content}"
    metadata_template = "{key}: {value},"   
    metadata_seperator= " "

    for doc in documents:
        doc.text_template = text_template
        doc.metadata_template = metadata_template
        doc.metadata_seperator = metadata_seperator
    
    print(documents[0].get_content(metadata_mode=MetadataMode.ALL))

    # Hide the File Name from the LLM
    documents[0].excluded_llm_metadata_keys = ["File Name", "Content Type", "Header Path"]
    #print(documents[0].get_content(metadata_mode=MetadataMode.LLM))

    # Hide the File Name from the embedding model
    documents[0].excluded_embed_metadata_keys = ["File Name"]
    #print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

    return documents


def create_vector_index_locally(data_repo:str, documents:List[Document]):

    path_storage = PERSIST_DIR + data_repo
    documents = None
    index = None

    # check if storage already exists
    if not os.path.exists(path_storage):
        documents = load_markdown_docs(data_repo)
        # load the documents and create the index
        index = VectorStoreIndex.from_documents(documents)
        # store locally for later use
        index.storage_context.persist(persist_dir=path_storage)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=path_storage)
        index = load_index_from_storage(storage_context)

    return index

def query(index,llm):
     # either way we can now query the index
    service_context = ServiceContext.from_defaults(llm=llm)
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query("What did the author do growing up?")
    
    print(response)

def load_data() -> VectorStoreIndex:
    """
    Load structured data (md file) using OpenAI llm and default inMemory storage
    """
    input_dir = SOURCE_DIR + "streamlit-docs"
    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    documents = reader.load_data()

    system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."
    llm = get_llm(1, "gpt-3.5-turbo",0.5,system_prompt=system_prompt)
    service_context = ServiceContext.from_defaults(llm=llm)
    vsindex = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return vsindex


def load_unstructured_data():
    """
    Load unstructured data into Pinecone using OpenAI llm
    """
    UnstructuredReader = download_loader("UnstructuredReader")

    input_dir = SOURCE_DIR + "llamindex-docs"

    dir_reader = SimpleDirectoryReader(
        input_dir = SOURCE_DIR + "llamindex-docs",
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
        llm = get_llm(1,"gpt-3.5-turbo",0)

        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002", embed_batch_size=100
        )
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser
        )
    else:

        # get Hugginface llm
        llm = get_llm(3,"",0)

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

    # llm = get_llm(1,"gpt-3.5-turbo",0.5,True)
    # response = llm.complete("Hi, say a 1 sentence poem")
    # print (response)
    
    # system_prompt="You are an expert Paul Graham's essay and your job is to answer literature questions. Assume that all questions are related to the Paul Graha's essay. Keep your answers based on facts - do not hallucinate features.",
    # load_data_tuto("tutorial",system_prompt)

    llm = get_llm(1,"gpt-3.5-turbo",0)
    service_context = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(service_context)

    # load documents from each folder. we keep them seperate for now, in order to create seperate indexes
    getting_started_docs = load_markdown_docs("llamindex-docs-tutorial/getting_started")
    gs_index = create_vector_index_locally("llamindex-docs-tutorial/getting_started",getting_started_docs)

