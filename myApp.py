import os
from dotenv import load_dotenv
import pinecone
from llama_index.readers import SimpleWebPageReader
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores import MilvusVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.postprocessor import (
    SimilarityPostprocessor,
    SentenceEmbeddingOptimizer,
)
from llama_index import SimpleDirectoryReader
import streamlit as st
import openai
from node_postprocessors.duplicate_postprocessing import (
    DuplicateRemoverNodePostprocessor,
)

load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
print(index_name)

EMBEDDING_SIZE = "1536"
debug_mode = True

llama_debug = LlamaDebugHandler(print_trace_on_end=debug_mode)
callback_manager = None

if debug_mode:
    callback_manager = CallbackManager(handlers=[llama_debug])


st.markdown("# APP DEMO")
st.sidebar.markdown("# Main")


def clear_cache():
    st.legacy_caching.caching.clear_cache()
    st.session_state.clear()


def scrapURL(url: str) -> VectorStoreIndex:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    # print(documents)
    index = VectorStoreIndex.from_documents(documents=documents)
    return index


def queryVSIndex(index: VectorStoreIndex, query: str):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)

    if debug_mode:
        print("\n\n\n*****DEBUG*****\n\n")
        print(llama_debug.get_llm_inputs_outputs())
        # Print info on llm inputs/outputs - returns start/end events for each LLM call
        event_pairs = llama_debug.get_llm_inputs_outputs()
        print("\nWhat was sent to LLM:\n")
        print(event_pairs[0][0])  # Shows what was sent to LLM
        print("\nPayload keys:\n")
        print(
            event_pairs[0][1].payload.keys()
        )  # What other things you can debug by simply passing the key
        print("\nLLM Response\n")
        print(
            event_pairs[0][1].payload["response"]
        )  # Shows the LLM response it generated.


# @st.cache_resource(show_spinner=False)
def get_vsindex(indexname: str, service_context: ServiceContext) -> VectorStoreIndex:
    index_name = indexname
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    pinecone_index = pinecone.Index(index_name=index_name)
    vectore_store = PineconeVectorStore(pinecone_index=pinecone_index)
    vsindex = VectorStoreIndex.from_vector_store(
        vector_store=vectore_store, service_context=service_context
    )

    return vsindex


@st.cache_resource(show_spinner=False)
def load_data():
    # with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
    #     reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    #     docs = reader.load_data()
    #     service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
    #     index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    #     return index
    pass


vsindex = load_data()


# st.set_page_config(
#     page_title="Chat with LlamaIndex Docs",
#     page_icon=":robot:",
#     layout = "centered",
#     initial_sidebar_state="auto",
#     menu_items=None,
# )
st.title("Chat with LlamaIndex Docs 💬 📚")
st.info("Check out the full tutorial", icon="📃")


print("Hello World LlamaIndex Demo")

st.cache_resource.clear()
st.cache_data.clear()


# index = scrapURL(url='https://cbarkinozer.medium.com/an-overview-of-the-llamaindex-framework-9ee9db787d16')
# queryVSIndex(index,"What is the purpose of the LLamaIndex framework?")

llamai_service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

vsindex = get_vsindex(index_name, service_context=llamai_service_context)


# query="What is a Llamaindex query engine?"
# query="What is a llamaindex agent?"
# queryVSIndex(vsindex,query)

if "chat_engine" not in st.session_state.keys():
    # Note that the processing process send API queries using the openAI embeding model => cost money
    sentenceemb_postprocessor = SentenceEmbeddingOptimizer(
        embed_model=llamai_service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

    similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)
    duplicaterem_postprocessor = DuplicateRemoverNodePostprocessor()

    node_postprocessors = [sentenceemb_postprocessor, duplicaterem_postprocessor]

    st.session_state.chat_engine = vsindex.as_chat_engine(
        chat_mode="context",
        verbose=True,
        node_postprocessors=[],
    )

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, ask me a quetion about LlamaIndex's open source Library!",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Q for debug: What is a llamaindex agent?

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
            st.write(response.response)

            nodes = [node for node in response.source_nodes]

            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.divider()
                    st.write(f"Source Node {i +1}: score= {node.score}")
                    st.write(node.metadata)
                    st.write(node.text)
                    st.divider()
