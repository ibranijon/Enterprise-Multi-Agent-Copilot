from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model="llama3.1:latest", temperature=0)

PLANNER_SYSTEM = """
You are the Planner Agent for a healthcare enterprise copilot.

Goal:
Generate document-retrieval tasks to search a small text document corpus (e.g., guidelines, reports, articles),
NOT a clinical database.

FORMAT (strict):
- One task per line
- Plain text only
- No numbering, no bullets, no labels, no headings

TASK RULES:
- Tasks must be phrased as document search intents (e.g., "Readmission drivers for heart failure", "Interventions reducing readmissions").
- Do NOT request patient records, charts, cohorts, time windows, counts, SQL, or "retrieve all patients".
- Do NOT assume access to EHR/EMR data or internal hospital systems.
- Generate only what is necessary (prefer 2–4 tasks), max 5.
"""
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM),
        ("human", "User task:\n{question}\n\nReturn the retrieval tasks:"),
    ]
)


def _to_list(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tasks = lines[:5]

    if not tasks:
        tasks = [
            "Retrieve healthcare documents relevant to the user's request, including definitions, risks, and mitigation strategies."
        ]

    return tasks


planner_agent = planner_prompt | llm | StrOutputParser() | RunnableLambda(_to_list)
