from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from config.api_keys import  APIKeysConfig

from utils.retrievers import (
    build_keyword_retriever,
    build_parent_document_retriever,
    create_splitters,
    initialize_vectorstore,
    build_ensemble_retriever
)

from data_models.models import GraphState
from utils.utils import load_and_preprocess_documents
from utils.chain import (
    get_generate_queries_chain,
    get_rag_chain,
    get_retrieval_grader,
    get_question_rewriter,
)

import warnings
warnings.filterwarnings("ignore", message=".*method='json_schema'.*")


llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()


docs = load_and_preprocess_documents('./data/pdf.pdf')
parent_splitter, child_splitter = create_splitters()


vectorstore = initialize_vectorstore(docs, embeddings)
parent_retriever = build_parent_document_retriever(vectorstore, parent_splitter, child_splitter)
keyword_retriever = build_keyword_retriever(docs)
ensemble_retriever = build_ensemble_retriever(parent_retriever, keyword_retriever)
generate_queries = get_generate_queries_chain(llm)
rag_chain = get_rag_chain(llm)


retrieval_grader = get_retrieval_grader(llm)
question_rewriter = get_question_rewriter(llm)
web_search_tool = TavilySearch(max_results=3)


# Retrieve documents using the question in state
def retrieve(state):
    print("RETRIEVE: Retrieving documents from vectorstore...")
    question = state["question"]
    retrieved_documents = ensemble_retriever.invoke(question)
    return {"documents": retrieved_documents, "question": question}

# Generate an answer using the retrieved documents and question
def generate(state):
    print("GENERATE: Generating response using retrieved documents and question...")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Filter documents based on relevance to the question
def grade_documents(state):
    print("GRADE: Checking document relevance to question...")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        relevance = score.relevance_score
        print(f"The relevance score for this document is: {relevance}")
        if relevance >= 50:
            print("\tGRADE: Document is relevant")
            filtered_docs.append(d)
        else:
            print("\tGRADE: Document is NOT relevant")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

# Rewrite the question to improve retrieval quality
def transform_query(state):
    print("TRANSFORM QUERY: Rewriting the query...")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

# Perform web search and append results to existing documents
def web_search(state):
    print("WEB SEARCH: Performing web search...")
    question = state["question"]
    documents = state["documents"]
    web_docs = web_search_tool.invoke({"query": question})
    print(web_docs)
    results = web_docs.get("results", [])
    if not results:
        print("No web search results found.")
        return {"documents": documents, "question": question}

    web_results = "\n".join([res.get("content", "") for res in results if "content" in res])
    if web_results:
        documents.append(Document(page_content=web_results))
    return {"documents": documents, "question": question}

# Decide whether to generate an answer or transform the query
def decide_to_generate(state):
    print("DECISION: Assessing graded documents...")
    web_search_flag = state["web_search"]

    if web_search_flag == "Yes":
        print("\tDECISION: All documents NOT relevant, transforming query")
        return "transform_query"
    else:
        print("\tDECISION: Documents relevant, generating response")
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()