from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    question: str
    plan: List[str]
    documents: List[Document]
    document_relevancy: bool
    generation: str
    retries: int
