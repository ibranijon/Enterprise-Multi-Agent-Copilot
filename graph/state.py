from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str
    plan: List[str]
    documents: List[Document]
    document_relevancy: bool
    research_trace: Dict[str, Any]

    writer_draft: Dict[str, Any]

    executive_summary: str
    email_to: str
    email_subject: str
    email_body: str
    email: str
    actions: List[Dict[str, Any]]
    citations_used: List[int]
    sources: str
    verifier_issues: List[str]


    final_output: str
    run_id: str
    trace: List[Dict[str, Any]]
