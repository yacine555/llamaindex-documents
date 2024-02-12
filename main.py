import os
import time
import asyncio
from dotenv import load_dotenv

from backend.eval import(
    generate_eval_questions
)

from backend.llm import get_llm
from backend.ingestion import (
    load_markdown_docs,
    create_vector_index_locally
)
from backend.eval import (
    evaluate_query_engine
)

from llama_index import (
    download_loader,
    load_index_from_storage,
    Response,
    ServiceContext,
    set_global_service_context,
    SimpleDirectoryReader,
)

from llama_index.evaluation import DatasetGenerator
from llama_index.evaluation import ResponseEvaluator
from llama_index.tools import QueryEngineTool
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

def run_test():

    llm = get_llm(1,"gpt-3.5-turbo",0)
    service_context = ServiceContext.from_defaults(llm=llm)
    # set_global_service_context(service_context)

    # load our documents from each folder. we keep them seperate for now, in order to create seperate indexes
    getting_started_docs = load_markdown_docs("llamindex-docs-tutorial/getting_started")
    gs_index = create_vector_index_locally("llamindex-docs-tutorial/getting_started",getting_started_docs)

    # community_docs = load_markdown_docs("llamindex-docs-tutorial/community")
    # community_index = create_vector_index_locally("llamindex-docs-tutorial/community",community_docs)

    # data_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/data_modules")
    # data_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/data_modules",data_docs)

    # agent_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/agent_modules")
    # agent_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/agent_modules",agent_docs)

    # model_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/model_modules")
    # model_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/model_modules",model_docs)

    # query_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/query_modules")
    # query_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/query_modules",query_docs)

    # supporting_docs = load_markdown_docs("llamindex-docs-tutorial/core_modules/supporting_modules")
    # supporting_index = create_vector_index_locally("llamindex-docs-tutorial/core_modules/supporting_modules",supporting_docs)

    # tutorials_docs = load_markdown_docs("llamindex-docs-tutorial/end_to_end_tutorials")
    # tutorials_index = create_vector_index_locally("llamindex-docs-tutorial/end_to_end_tutorials",tutorials_docs)

    # contributing_docs = load_markdown_docs("llamindex-docs-tutorial/development")
    # contributing_index = create_vector_index_locally("llamindex-docs-tutorial/development",contributing_docs)
   

    # create a query engine tool for each index
    getting_started_tool = QueryEngineTool.from_defaults(
        query_engine=gs_index.as_query_engine(), 
        name="Getting Started", 
        description="Useful for answering questions about installing and running llama index, as well as basic explanations of how llama index works."
    )

    # community_tool = QueryEngineTool.from_defaults(
    #     query_engine=community_index.as_query_engine(),
    #     name="Community",
    #     description="Useful for answering questions about integrations and other apps built by the community."
    # )

    # data_tool = QueryEngineTool.from_defaults(
    #     query_engine=data_index.as_query_engine(),
    #     name="Data Modules",
    #     description="Useful for answering questions about data loaders, documents, nodes, and index structures."
    # )

    # agent_tool = QueryEngineTool.from_defaults(
    #     query_engine=agent_index.as_query_engine(),
    #     name="Agent Modules",
    #     description="Useful for answering questions about data agents, agent configurations, and tools."
    # )

    # model_tool = QueryEngineTool.from_defaults(
    #     query_engine=model_index.as_query_engine(),
    #     name="Model Modules",
    #     description="Useful for answering questions about using and configuring LLMs, embedding modles, and prompts."
    # )

    # query_tool = QueryEngineTool.from_defaults(
    #     query_engine=query_index.as_query_engine(),
    #     name="Query Modules",
    #     description="Useful for answering questions about query engines, query configurations, and using various parts of the query engine pipeline."
    # )

    # supporting_tool = QueryEngineTool.from_defaults(
    #     query_engine=supporting_index.as_query_engine(),
    #     name="Supporting Modules",
    #     description="Useful for answering questions about supporting modules, such as callbacks, service context, and avaluation."
    # )

    # tutorials_tool = QueryEngineTool.from_defaults(
    #     query_engine=tutorials_index.as_query_engine(),
    #     name="Tutorials",
    #     description="Useful for answering questions about end-to-end tutorials and giving examples of specific use-cases."
    # )

    # contributing_tool = QueryEngineTool.from_defaults(
    #     query_engine=contributing_index.as_query_engine(),
    #     name="Contributing",
    #     description="Useful for answering questions about contributing to llama index, including how to contribute to the codebase and how to build documentation."
    # )

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            getting_started_tool,
            # community_tool,
            # data_tool,
            # agent_tool,
            # model_tool,
            # query_tool,
            # supporting_tool,
            # tutorials_tool,
            # contributing_tool
        ],
        # enable this for streaming
        # response_synthesizer=get_response_synthesizer(streaming=True),
        verbose=False
    )

    # print("=================================")
    # print("EVAL QUERY ENGINE TEST:")
    # response = query_engine.query("How do I install llama index?")
    # print(str(response))
    # print("=================================")

    print("=================================")
    print("GENERATE QUESTIONS")
    question_dataset = generate_eval_questions("llamindex-docs-tutorial/getting_started/")
    print("=================================")


    print("=================================")
    print("EVAL QUESTIONS")
    # llm = get_llm(1,"gpt-4",0)
    llm2 = get_llm(1,"gpt-3.5-turbo",0)
    service_context2 = ServiceContext.from_defaults(llm=llm2)
    evaluator = ResponseEvaluator(service_context=service_context)
    all_results = evaluate_query_engine(evaluator, query_engine, question_dataset)
    print("=================================")

    # total_correct, all_results = evaluate_query_engine(evaluator, query_engine, question_dataset)
    # print(f"Halucination? Scored {total_correct} out of {len(question_dataset)} questions correctly.")

   
    total_correct = 0
 
    # for r in all_results:
    #     print(r)
    #     total_correct += 1
    # print(f"Halucination? Scored {total_correct} out of {len(all_results)} questions correctly.")



if __name__ == "__main__":
    run_test()
    # asyncio.run(run_test())