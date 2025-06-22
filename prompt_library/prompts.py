from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import hub

prompt_from_hub = hub.pull("rlm/rag-prompt")


multiple_queries_prompt = ChatPromptTemplate(input_variables=['original_query'],
                            messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],template='You are a helpful assistant that generates multiple search queries based on a single input query.')),
                            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['original_query'], template='Generate multiple search queries related to: {question} \n OUTPUT (3 queries):'))])


system_grade = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grade),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

system_rewrite = """You are a question re-writer. Your task is to rewrite the input question to a clearer, more specific, and search-optimized form without changing its meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the original question:\n\n{question}\n\nRewrite the question clearly and concisely for better web search results.",
        ),
    ]
)
