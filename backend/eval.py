import os
import random
random.seed(42)
import time
import asyncio
from dotenv import load_dotenv

from .llm import get_llm
from .ingestion import (
    load_markdown_docs,
    create_vector_index_locally
)

from llama_index.prompts import (
    Prompt,
    PromptTemplate
)

from llama_index.evaluation import DatasetGenerator
from llama_index.evaluation import ResponseEvaluator
from llama_index.tools import QueryEngineTool
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer


from data.markdown_docs_reader import MarkdownDocsReader
from llama_index import (
    download_loader,
    load_index_from_storage,
    Response,
    ServiceContext,
    set_global_service_context,
    SimpleDirectoryReader,
)

# Make the printing look nice
from llama_index.schema import (
    Document,
    MetadataMode
)

SOURCE_DIR = "./data/source/"
PERSIST_DIR = "./data/storage/"

def generate_eval_questions(filepath:str):

    filepath = SOURCE_DIR + filepath
    documents = SimpleDirectoryReader(filepath, recursive=True, required_exts=[".md"]).load_data()
    all_text = ""

    for doc in documents:
        all_text += doc.text
    giant_document = Document(text=all_text)

    llm = get_llm(1,"gpt-3.5-turbo",0)
    gpt4_service_context = ServiceContext.from_defaults(llm=llm)

    question_dataset = []
    if os.path.exists(filepath + "question_dataset.txt"):
        print("Loading existing questions...")
        with open(filepath + "question_dataset.txt", "r") as f:
            for line in f:
                question_dataset.append(line.strip())
    else:
        # generate questions
        print("Generating questions for evaluation...")
        data_generator = DatasetGenerator.from_documents(
            [giant_document],
            text_question_template=Prompt(
                "A sample from the LlamaIndex documentation is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Using the documentation sample, carefully follow the instructions below:\n"
                "{query_str}"
            ),
            question_gen_query=(
                "You are an evaluator for a search pipeline. Your task is to write a single question "
                "using the provided documentation sample above to test the search pipeline. The question should "
                "reference specific names, functions, and terms. Restrict the question to the "
                "context information provided.\n"
                "Question: "
            ),
            # set this to be low, so we can generate more questions
            service_context=gpt4_service_context
        )
        generated_questions = data_generator.generate_questions_from_nodes()
        
        # randomly pick 40 questions from each dataset
        print(generated_questions)
        # generated_questions = random.sample(generated_questions, 40)
        question_dataset.extend(generated_questions)

        print(f"Generated {len(question_dataset)} questions.")

        # save the questions!
        with open(filepath + "question_dataset.txt", "w") as f:
            for question in question_dataset:
                f.write(f"{question.strip()}\n")

    return question_dataset

def evaluate_query_engine(evaluator, query_engine, questions):
    async def run_query(query_engine, q):
        try:
            return await query_engine.aquery(q)
        except:
            return Response(response="Error, query failed.")

    total_correct = 0
    all_results = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size:batch_size+5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")

        for response in responses:
            eval_result = 1 if "YES" in evaluator.evaluate(response) else 0
            total_correct += eval_result
            all_results.append(eval_result)
        
        # helps avoid rate limits
        time.sleep(1)

    return total_correct, all_results


async def run_query(query_engine, q):
    try:
        return await query_engine.aquery(q)
    except:
        return Response(response="Error, query failed.")

async def run_query2(query_engine, q):
    return await query_engine.aquery(q)

async def query_batch_question(query_engine, batch_qs):
    tasks = [run_query2(query_engine, q) for q in batch_qs]
    L = await asyncio.gather(*tasks)
    print(L)
    return L


def evaluate_query_engine(evaluator, query_engine, questions):

    print(f"Start Eval on {len(questions)} questions ...")
    total_correct = 0
    all_results = []
    step = 2


    q1 = "What are the key concepts and modules in LlamaIndex for composing a Retrieval Augmented Generation (RAG) pipeline?"
    q2 = "What is the purpose of the `SimpleDirectoryReader` function in the LlamaIndex documentation?"

    print(f"Eval question: {q1}")
    response1 = query_engine.query(q1)
    print(response1)
    print(f"Eval question: {q2}")
    response2 = query_engine.query(q2)
    print(response2)

    for batch_size in range(0, len(questions), step):
        batch_qs = questions[batch_size:batch_size+step]
        print(f"batch index start: {batch_size},  question: {batch_qs}")

        # tasks = [run_query(query_engine, q) for q in batch_qs]

        # responses = asyncio.run(query_batch_question(query_engine,batch_qs))

        print(f"Eval question: {batch_qs[0]}")
        response1 = query_engine.query(batch_qs[0])
        print(response1)
        response2 = query_engine.query(batch_qs[1])
        print(response2)

        print(f"finished batch {(batch_size // step) + 1} out of {len(questions) // step}")


        # for response in responses:
        #     eval_result = 1 if "YES" in evaluator.evaluate(response) else 0
        #     total_correct += eval_result
        #     all_results.append(eval_result)
        #     print(f"Eval q res: {eval_result}")
        
        # helps avoid rate limits with OpenAI API
        time.sleep(1)

    # return total_correct, all_results
    return all_results



if __name__ == "__main__":

    llm = get_llm(1,"gpt-3.5-turbo",0)
    service_context = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(service_context)

    # load our documents from each folder. we keep them seperate for now, in order to create seperate indexes
    getting_started_docs = load_markdown_docs("llamindex-docs-tutorial/getting_started")
    gs_index = create_vector_index_locally("llamindex-docs-tutorial/getting_started",getting_started_docs)

    community_docs = load_markdown_docs("llamindex-docs-tutorial/community")
    community_index = create_vector_index_locally("llamindex-docs-tutorial/community",community_docs)

    data_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/data_modules")
    data_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/data_modules",data_docs)

    agent_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/agent_modules")
    agent_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/agent_modules",agent_docs)

    model_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/model_modules")
    model_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/model_modules",model_docs)

    query_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/query_modules")
    query_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/query_modules",query_docs)

    supporting_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/supporting_modules")
    supporting_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/supporting_modules",supporting_docs)

    tutorials_docs = load_markdown_docs("llamindex-docs-tutorial/end_to_end_tutorials")
    tutorials_index = create_vector_index_locally("llamindex-docs-tutorial/end_to_end_tutorials",tutorials_docs)

    contributing_docs = load_markdown_docs("llamindex-docs-tutorial/development")
    contributing_index = create_vector_index_locally("llamindex-docs-tutorial/development",contributing_docs)
   

    # create a query engine tool for each folder
    getting_started_tool = QueryEngineTool.from_defaults(
        query_engine=gs_index.as_query_engine(), 
        name="Getting Started", 
        description="Useful for answering questions about installing and running llama index, as well as basic explanations of how llama index works."
    )

    community_tool = QueryEngineTool.from_defaults(
        query_engine=community_index.as_query_engine(),
        name="Community",
        description="Useful for answering questions about integrations and other apps built by the community."
    )

    data_tool = QueryEngineTool.from_defaults(
        query_engine=data_index.as_query_engine(),
        name="Data Modules",
        description="Useful for answering questions about data loaders, documents, nodes, and index structures."
    )

    agent_tool = QueryEngineTool.from_defaults(
        query_engine=agent_index.as_query_engine(),
        name="Agent Modules",
        description="Useful for answering questions about data agents, agent configurations, and tools."
    )

    model_tool = QueryEngineTool.from_defaults(
        query_engine=model_index.as_query_engine(),
        name="Model Modules",
        description="Useful for answering questions about using and configuring LLMs, embedding modles, and prompts."
    )

    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_index.as_query_engine(),
        name="Query Modules",
        description="Useful for answering questions about query engines, query configurations, and using various parts of the query engine pipeline."
    )

    supporting_tool = QueryEngineTool.from_defaults(
        query_engine=supporting_index.as_query_engine(),
        name="Supporting Modules",
        description="Useful for answering questions about supporting modules, such as callbacks, service context, and avaluation."
    )

    tutorials_tool = QueryEngineTool.from_defaults(
        query_engine=tutorials_index.as_query_engine(),
        name="Tutorials",
        description="Useful for answering questions about end-to-end tutorials and giving examples of specific use-cases."
    )

    contributing_tool = QueryEngineTool.from_defaults(
        query_engine=contributing_index.as_query_engine(),
        name="Contributing",
        description="Useful for answering questions about contributing to llama index, including how to contribute to the codebase and how to build documentation."
    )

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            getting_started_tool,
            community_tool,
            data_tool,
            agent_tool,
            model_tool,
            query_tool,
            supporting_tool,
            tutorials_tool,
            contributing_tool
        ],
        # enable this for streaming
        # response_synthesizer=get_response_synthesizer(streaming=True),
        verbose=False
    )


    print("EVAL QUERY ENGINE:")
    response = query_engine.query("How do I install llama index?")
    print(str(response))

    question_dataset = generate_eval_questions("llamindex-docs-tutorial/getting_started")

    llm = get_llm(1,"gpt-4",0)
    service_context = ServiceContext.from_defaults(llm=llm)
    elvaluator = ResponseEvaluator(service_context=service_context)

    total_correct, all_results = evaluate_query_engine(elvaluator, query_engine, question_dataset)

    print(f"Halucination? Scored {total_correct} out of {len(question_dataset)} questions correctly.")
