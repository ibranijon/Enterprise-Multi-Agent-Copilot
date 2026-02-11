from __future__ import annotations

from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


VERIFIER_SYSTEM = """
You are the Verifier Agent for an enterprise copilot.

Input:
- User request
- Writer draft JSON
- Evidence subset: ONLY the chunks referenced by citations_used

Tasks:
1) Relevancy: verify the draft addresses the user request.
2) Grounding: verify key claims and action items are supported by the evidence subset.
   - It does not need to use all sources.
   - It must be supported by at least one cited chunk, otherwise it is unsupported.
3) If unsupported:
   - Revise by removing/hedging unsupported claims, OR
   - Mark as "Not found in sources."
4) Add confidence per action item:
   - high: explicitly supported by evidence subset
   - medium: reasonable inference from evidence subset
   - low: weak or not found in sources (must be flagged)
5) If the draft cannot be grounded enough to be useful, output invalid.

Hard rules:
- Do not invent citations. Do not add new citations_used indices.
- Keep Executive Summary <= 150 words.
- Return ONLY valid JSON matching the schema.
"""


class VerifiedActionItem(BaseModel):
    task: str
    owner: str
    due_date: str
    confidence: str = Field(..., description="high|medium|low")
    confidence_rationale: str


class VerifierOutput(BaseModel):
    is_relevant: bool
    is_grounded: bool
    issues: List[str]
    executive_summary: str
    email_to: str
    email_subject: str
    email_body: str
    actions: List[VerifiedActionItem]
    citations_used: List[int]
    invalid: str | None = None


def _format_cited_evidence(docs: List[Document], cited_nums: List[int]) -> str:
    parts: List[str] = []
    for n in cited_nums or []:
        idx = n - 1
        if idx < 0 or idx >= len(docs):
            continue
        d = docs[idx]
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        md = d.metadata or {}
        src = md.get("source", "unknown")
        p = md.get("page_start", md.get("page"))
        c = md.get("chunk_id", "?")
        parts.append(f"[{n}] ({src} | page {p} | chunk {c})\n{txt}")
    return "\n\n".join(parts).strip()


def build_verifier_agent():
    parser = JsonOutputParser(pydantic_object=VerifierOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFIER_SYSTEM),
            (
                "human",
                "USER REQUEST:\n{question}\n\n"
                "WRITER DRAFT JSON:\n{draft_json}\n\n"
                "CITED EVIDENCE SUBSET:\n{evidence}\n\n"
                "{format_instructions}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def verify_draft(question: str, documents: List[Document], draft: Dict[str, Any]) -> Dict[str, Any]:
    if "invalid" in draft:
        return {
            "invalid": draft["invalid"],
            "is_relevant": False,
            "is_grounded": False,
            "issues": ["Insufficient evidence: no documents provided."],
            "citations_used": [],
            "executive_summary": "",
            "email_to": "",
            "email_subject": "",
            "email_body": "",
            "actions": [],
        }

    cited_nums = draft.get("citations_used", []) or []
    evidence = _format_cited_evidence(documents, cited_nums)

    if not evidence.strip():
        return {
            "invalid": "Invalid: I cannot answer your question because of insufficient data in the provided sources.",
            "is_relevant": False,
            "is_grounded": False,
            "issues": ["No cited evidence available to verify grounding."],
            "citations_used": cited_nums,
            "executive_summary": "",
            "email_to": "",
            "email_subject": "",
            "email_body": "",
            "actions": [],
        }

    chain = build_verifier_agent()
    return chain.invoke(
        {
            "question": question,
            "draft_json": draft,
            "evidence": evidence,
        }
    )
