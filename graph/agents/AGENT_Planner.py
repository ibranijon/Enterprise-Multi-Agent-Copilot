from typing import Any, Dict, List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

llm = ChatOllama(model="llama3.1:latest", temperature=0)

PLANNER_SYSTEM = """You are the Planner Agent for a healthcare enterprise copilot.

Goal:
Decompose the user's task into a short list of retrieval tasks that the Research step will execute.

Rules:
- Output ONLY bullet points, each starting with "- ".
- Maximum 5 bullets, minimum 1 bullet.
- Each bullet must be a concrete search/retrieval instruction (what to look for in documents).
- Do NOT invent facts.
- Do NOT ask questions unless absolutely necessary.
- Assume the task is healthcare-related and ALWAYS requires retrieval/evidence.
- Bullets should be phrased so they can be used as search queries (include key terms from the user request).
"""

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM),
        ("human", "User task:\n{question}\n\nReturn the retrieval plan as bullet points:"),
    ]
)

def _normalize_plan(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    bullets: List[str] = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            item = ln.lstrip("-•* ").strip()
        else:
            item = ln.strip()

        if item:
            bullets.append("- " + item)

    # enforce 1..5 bullets
    bullets = bullets[:5]
    if not bullets:
        bullets = ["- Retrieve evidence directly relevant to the user's task (definitions, requirements, constraints, and any cited metrics)."]

    return "\n".join(bullets)

planner_agent: RunnableSequence = (
    planner_prompt | llm | StrOutputParser() | RunnableLambda(_normalize_plan))
