from langchain_core.output_parsers import StrOutputParser
from prompt_library.prompts import multiple_queries_prompt, grade_prompt, prompt_from_hub, re_write_prompt
from data_models.models import GradeDocuments


def get_generate_queries_chain(llm):
    return multiple_queries_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))


def get_rag_chain(llm):
    return prompt_from_hub | llm | StrOutputParser()


def get_retrieval_grader(llm):
    structured = llm.with_structured_output(GradeDocuments)
    return grade_prompt | structured


def get_question_rewriter(llm):
    return re_write_prompt | llm | StrOutputParser()
