from __future__ import annotations


from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import date, timedelta

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
from langchain_openai import ChatOpenAI




WRITER_SYSTEM = """
You are the Writer Agent for a healthcare enterprise copilot.

You receive:
- The user's request
- A set of retrieved evidence chunks (documents)

Your job:
1) Write an Executive Summary (MAX 150 words).
2) Write a client-ready email (professional, approachable).

Critical rules:
- Use ONLY the minimum subset of chunks needed to answer well.
- It is OK to IGNORE chunks that are not needed.
- Do NOT try to fit all chunks into the response.
- Do NOT add medical advice beyond the evidence. No unsupported claims.
- If evidence is insufficient, say so clearly.

Citations:
- You MUST include a list of citation indices you actually used (1-based indices into the provided documents list).
- Only cite what you used. Do NOT cite unused chunks.

Action List:
- Create 2-4 actionable items.
- Owner: if user provides an email in the request, use it; otherwise use "[EMAIL]".
- Due date: if user provides it, use it; otherwise choose a realistic due date soon.
- Confidence: "high" | "medium" | "low" based on evidence coverage.

Output format:
Return ONLY valid JSON matching the schema.
"""


class ActionItem(BaseModel):
    task: str = Field(..., min_length=3)
    owner: str = Field(..., min_length=3)
    due_date: str = Field(..., description="YYYY-MM-DD")
    confidence: str = Field(..., description="high|medium|low")


class WriterModelOutput(BaseModel):
    executive_summary: str
    email_subject: str
    email_to: str
    email_body: str
    actions: List[ActionItem]
    citations_used: List[int]


def _extract_email(text: str) -> Optional[str]:
    import re
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return m.group(0) if m else None


def _format_docs_for_prompt(docs: List[Document]) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        md = d.metadata or {}
        src = md.get("source", "unknown")
        p = md.get("page_start", md.get("page"))
        c = md.get("chunk_id", "?")
        parts.append(f"[{i}] ({src} | page {p} | chunk {c})\n{txt}")
    return "\n\n".join(parts).strip()


def _default_due_date() -> str:
    # realistic near-term due date
    return (date.today() + timedelta(days=7)).isoformat()


def build_writer_agent(model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name,temperature=0)

    parser = JsonOutputParser(pydantic_object=WriterModelOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM),
            (
                "human",
                "USER REQUEST:\n{question}\n\nEVIDENCE CHUNKS:\n{context}\n\n{format_instructions}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain


def writer_agent(question: str, documents: List[Document], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    # Failure behavior: insufficient evidence
    if not documents:
        to_email = _extract_email(question) or "[EMAIL]"
        due = _default_due_date()
        return {
            "executive_summary": "Insufficient evidence in the provided sources to answer the request.",
            "email": f"To: {to_email}\nSubject: Heart failure readmissions – insufficient evidence\n\n"
                     "Hi,\n\nI reviewed the available sources, but they do not contain enough relevant evidence to answer this request reliably.\n\n"
                     "Best regards,",
            "actions": [
                {"task": "Provide additional documents or guidelines specific to the question.", "owner": to_email, "due_date": due, "confidence": "low"}
            ],
            "sources": "",
        }

    chain = build_writer_agent(model_name=model_name)

    context = _format_docs_for_prompt(documents)
    raw: Dict[str, Any] = chain.invoke({"question": question, "context": context})

    # Post-process: enforce email_to placeholder rule
    user_email = _extract_email(question)
    email_to = raw.get("email_to") or user_email or "[EMAIL]"
    if user_email:
        email_to = user_email

    # Ensure due dates exist (fallback if model forgets)
    actions = raw.get("actions", [])
    for a in actions:
        if not a.get("owner"):
            a["owner"] = email_to
        if not a.get("due_date"):
            a["due_date"] = _default_due_date()

    # Build sources section using ONLY citations_used
    sources_block = format_sources_block_from_metadata(documents, raw.get("citations_used", []))

    # Final email string (client-ready)
    email = f"To: {email_to}\nSubject: {raw.get('email_subject','Update')}\n\n{raw.get('email_body','').strip()}"

    return {
        "executive_summary": raw.get("executive_summary", "").strip(),
        "email": email,
        "actions": actions,
        "sources": sources_block,
    }


def format_sources_block_from_metadata(docs: List[Document], cited_nums: List[int]) -> str:
    # cited_nums are 1-based indices
    if not cited_nums:
        return ""

    seen = set()
    lines = ["Sources"]
    for n in cited_nums:
        if n in seen:
            continue
        seen.add(n)

        idx = n - 1
        if idx < 0 or idx >= len(docs):
            continue

        md = docs[idx].metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page_start", md.get("page"))
        chunk = md.get("chunk_id", "?")

        lines.append(f"- {src} (page {page}, chunk {chunk})")

    return "\n".join(lines)
