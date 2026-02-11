from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model="llama3.1:latest", temperature=0)

PLANNER_SYSTEM = """
You are the Planner Agent for a healthcare enterprise copilot.
Your task:
Decompose the user's request into ONLY the minimum number of retrieval tasks required
to answer the request accurately.

- Output each task on its own line.
- Output ONLY plain text lines.
- DO NOT use numbering (e.g. "1.", "2)", "Step 1").
- DO NOT use bullet symbols ("-", "*", "•").
- DO NOT add headings, explanations, or extra text.
- Each line must begin directly with the task text.
- Generate ONLY the tasks that are strictly necessary.
- Minimum 1 task, maximum 5 tasks.
- Each task must be a concrete, actionable retrieval instruction suitable for searching documents.

If you violate any formatting rule, the output is incorrect.
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
