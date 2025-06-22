from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GradeDocuments(BaseModel):
    relevance_score: float = Field(
        description="Relevance score of the document to the question, from 0 to 100"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
