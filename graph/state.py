from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    question: str
    plan: List[str]
    documents: List[Document]
    document_relevancy: bool
    research_trace: Dict[str, Any]
    generation: str
    
