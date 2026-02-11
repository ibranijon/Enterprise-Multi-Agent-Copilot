from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    question: str
    plan: str
    generation: str
    document_relevancy: bool
    documents: List[Document]
    retries: int
