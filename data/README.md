# Multi-Agent Healthcare Enterprise Copilot

## Overview

This project is a **multi-agent, evidence-grounded enterprise copilot** designed for healthcare use cases, with a primary focus on **heart failure readmissions, risk factors, and transitional care interventions**.

The system combines **retrieval‑augmented generation (RAG)** with **multiple specialized LLM agents** coordinated through a **graph‑based workflow**, producing structured, client‑ready outputs that are strictly grounded in retrieved evidence.

The copilot is intended for **research synthesis, clinical decision support discussions, and health systems analysis**. It does **not** provide patient‑specific medical advice.

---

## Key Technologies

- **LLM (Reasoning / Generation):** `gpt-4o-mini`
- **Embedding Model:** `text-embedding-3-small`
- **Vector Store:** ChromaDB
- **Frameworks:** LangChain, LangGraph
- **UI:** Streamlit
- **Evaluation:** Custom evaluation harness
- **Environment Management:** `uv`

---

## Dataset Summary

### General Source Overview

The dataset is a curated collection of **peer‑reviewed clinical research** focused on **heart failure (HF)** and **hospital readmissions**, particularly within **30, 90, and 180 days after discharge**. Sources include randomized controlled trials, cohort studies, systematic reviews, and health‑system evaluations.

Collectively, the documents examine:

- Why heart failure readmissions occur
- Which clinical, functional, and system‑level factors increase risk
- Which transitional care, rehabilitation, and medication‑management strategies reduce rehospitalization

The evidence is **population‑level and research‑based**, enabling explanation and comparison of interventions rather than individualized prediction.

### What the Dataset Is Used For

The dataset enables the copilot to:

- Explain **multi‑factorial causes** of HF readmissions
- Compare **risk factors and predictors**
- Summarize **evidence‑based interventions** (e.g., transitional care, self‑management, cardiac rehabilitation)
- Translate research findings into **structured summaries, emails, and action lists**

### Prompt Examples

When querying the system, prompts should be **analytical, evidence‑seeking, and role‑aware**.

**Example of a good prompt:**

> How do frailty and functional status compare in their ability to predict short-term readmission in older heart failure patients? Do a proper summary of the question at hand and draft the answer to the cardiology team at [cardio@gmail.com](mailto:cardio@gmail.com). Furthermore, give an action list to the cardiology team on what to do next.

> Why is left ventricular ejection fraction alone often insufficient for predicting hospital readmissions in heart failure? Do a proper summary of the question at hand and draft the answer to the Cardiovascular Department at [cardivascualrdepart@gmail.com](mailto:cardivascualrdepart@gmail.com). Furthermore, give an action list to the medical team on what to do next.

This type of prompt clearly specifies:

- The clinical quesiton
- The department/person who is being addressed to
- The email of the department/person mentioned

Note: The agent will create the action list based on which department it deems most suited for the task at hand.
Confidence displayed at the items in the action list is generated based on how much agents preplexity with the chunks retrived to form the question.

---

## Dataset Requirements

The dataset is included in the repository.

If you want to run the application with your own documents, place **PDF files** at the project root:

```text
/data
```

These documents are ingested and indexed during the one‑time ingestion step.

---

## High‑Level Workflow

### 1. Ingestion (One‑Time Step)

**File:** `ingestion.py`

- Reads PDF documents from `/data`
- Cleans and chunks text
- Generates embeddings using `text-embedding-3-small`
- Stores embeddings in **ChromaDB**

Run this step once, or whenever the dataset changes.

---

### 2. Query‑Time Multi‑Agent Workflow

The runtime workflow is implemented as a **LangGraph state machine**, where each agent is a dedicated node.

#### Planner Agent

- Decomposes the user query into **1–5 document‑retrieval sub‑tasks**
- Tasks are phrased strictly as search intents

#### Research Agent

- Retrieves evidence chunks using the document retriever
- Filters out irrelevant chunks using LLM‑based relevance checks

#### Writer Agent

- Produces a structured draft containing:
  - Executive Summary
  - Client‑ready Email
  - Action List

- Uses only retrieved evidence
- Outputs explicit citations

#### Verifier Agent

- Ensures task relevance and evidence grounding
- Rejects unsupported or unsafe requests
- Returns either a verified response or a single invalid sentence

---

## Graph‑Based Architecture

- Each agent has its own system prompt and node
- Nodes are connected via a deterministic graph flow
- State is shared and merged across agents

This design ensures:

- Clear separation of responsibilities
- Transparency and traceability
- Easy extensibility

---

## Logging & Tracing

Agent‑level logs capture:

- Execution steps
- Routing decisions
- Evidence usage

This supports debugging, auditing, and explainability.

---

## Guardrails & Safety

The system enforces strict guardrails:

- No patient‑specific predictions
- No personalized treatment plans
- No hallucinated citations
- Explicit rejection of unsupported requests

Final enforcement is handled by the **Verifier Agent**.

---

## Evaluation Framework

A dedicated evaluation setup is included:

```text
/eval
  ├── run_eval.py
  └── /test_prompts
        └── test_prompts.jsonl
```

The evaluation suite runs curated prompts through the full graph to validate correctness, grounding, and failure handling.
In order to run the evaluation set, run with:

```bash
uv run python -m eval.run_eval
```

---

## Streamlit User Interface

A lightweight UI is provided for interactive use.

```text
/app/streamlit_app.py
```

Run with:

```bash
uv run streamlit run app/streamlit_app.py
```

---

## Environment Setup

### Required Environment Variable

```bash
OPENAI_API_KEY=your_api_key_here
```

Set this value in an `.env` file at the project root.

---

## Installed Dependencies

```python
dependencies = [
    "black>=26.1.0",
    "chromadb>=1.5.0",
    "isort>=7.0.0",
    "langchain>=1.2.10",
    "langchain-chroma>=1.1.0",
    "langchain-community>=0.4.1",
    "langchain-openai>=1.1.9",
    "langgraph>=1.0.8",
    "pdfplumber>=0.11.9",
    "python-dotenv>=1.2.1",
    "streamlit>=1.54.0",
    "tiktoken>=0.12.0",
]
```

## Summary

This project delivers a **robust, transparent, and safety‑constrained healthcare copilot** that translates complex heart‑failure research into structured, actionable outputs—while remaining firmly grounded in evidence and bounded by clear limitations.

## Streamlit URL

Below you will the StreamLit URL, note that logger files are only accessible through the local deployment

https://enterprise-multi-agent-copilot-rw6qdkmsxu7nooaa7v3fxg.streamlit.app/
