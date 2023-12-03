import os
from dotenv import load_dotenv

from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    SimpleDirectoryReader,
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
import pinecone
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

"""
Dowload the Natural Toolkit resources on first use if not available
"""
#nltk.download()

load_dotenv()


def load_data()-> VectorStoreIndex:
    """
    Load structured data (md file) using OpenAI llm and default inMomory storage
    """
    reader = SimpleDirectoryReader(input_dir="./docs/streamlit-docs", recursive=True)
    docs = reader.load_data()

    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
    vsindex = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return vsindex
    
def load_unstructured_data():
    """
    Load unstructured data into Pinecon using OpenAI llm
    """
    UnstructuredReader = download_loader('UnstructuredReader')

    dir_reader = SimpleDirectoryReader('./docs/llamindex-docs', file_extractor={
    ".pdf": UnstructuredReader(),
    ".html": UnstructuredReader(),
    ".eml": UnstructuredReader(),
    })
    documents = dir_reader.load_data()
    #print(documents[0].text)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    #nodes = node_parser.get_nodes_from_documents(documents=documents)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model,node_parser=node_parser
    )

    index_name = os.environ["PINECONE_INDEX_NAME"]
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
    load_unstructured_data()

