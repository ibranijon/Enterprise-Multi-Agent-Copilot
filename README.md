# Multi-Agent Healthcare Enterprise Copilot

## Overview

This project is a multi-agent, evidence-grounded enterprise copilot designed for healthcare use cases, with a primary focus on **heart failure readmissions, risk factors, and transitional care interventions**.

The system leverages:

- **Retrieval-Augmented Generation (RAG)**
- **Multiple specialized LLM agents**
- **A graph-based workflow (LangGraph)**
- **Strict grounding and verification rules**

Its goal is to produce **structured, client-ready outputs** (executive summaries, emails, and action lists) that are **explicitly grounded in retrieved evidence**, while safely rejecting requests that cannot be supported by the available data.

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

## Dataset Requirements

**The dataset is included in the repository.**

However if you want to run the application with your own dataset, you must modify the following folder at the project root:

```text
/data
```

Place your **PDF documents** (peer-reviewed papers, reports, guidelines) inside this folder.

These documents are used to build the vector database during ingestion.

---

## High-Level Workflow

### 1. Ingestion (One-Time Step)

**File:** `ingestion.py`

- Reads PDF documents from `/data`
- Cleans and chunks text
- Chunks are embedded using `text-embedding-3-small`
- Embeddings are stored in **ChromaDB**

This step is run **once**, or whenever the dataset changes.

---

### 2. Query-Time Multi-Agent Workflow

The runtime workflow is implemented as a **LangGraph state machine**, where each agent is represented by its own node.

#### Step-by-step Flow

1. **Planner Agent**
   - Takes the user’s query
   - Decomposes it into **1–5 document-retrieval sub-tasks**
   - Tasks are phrased strictly as _document search intents_

2. **Research Agent**
   - Uses `doc_retriever.py` to retrieve **2 chunks per sub-task**
   - For each retrieved chunk:
     - Calls `gpt-4o-mini` to validate whether the chunk is **directly relevant**
     - Irrelevant chunks are dropped

   - Result: a filtered, grounded evidence set

3. **Writer Agent**
   - Produces a **draft deliverable** consisting of:
     - Executive Summary (≤150 words)
     - Client-ready Email
     - Action List (2–4 items)

   - Uses **only a subset of retrieved chunks**
   - Outputs `citations_used` (1-based indices)

4. **Verifier Agent**
   - Validates the Writer’s draft by checking:
     - Task relevance (did it actually answer the user’s request?)
     - Grounding (are claims supported by cited evidence?)

   - Adds **confidence levels** to action items
   - If the request cannot be supported:
     - Returns a **single-sentence invalid response**
     - No partial or hallucinated output is allowed

5. **End Output**
   - Either:
     - A fully structured, verified response
     - OR a single invalid sentence indicating insufficient data

---

## Graph-Based Architecture

- Each agent has:
  - Its own **system prompt**
  - Its own **node**

- Nodes are connected via a **graph flow**
- State is shared and merged across nodes
- Execution always returns a final state object

This design ensures:

- Deterministic control flow
- Clear separation of responsibilities
- Easy extensibility

---

## Logging & Tracing

The project includes a `logger/` directory that captures:

- Agent-level execution steps
- Decisions made between nodes
- Traceability of how outputs were produced

Logs enable:

- Debugging
- Transparency
- Auditability of agent behavior

---

## Guardrails & Safety

Each agent’s **system prompt includes role-specific guardrails**, ensuring that:

- No patient-specific predictions are made
- No unsupported claims are generated
- No hallucinated citations appear
- Requests outside the dataset scope are rejected

The Verifier Agent enforces these rules at the final stage.

---

## Evaluation Framework

The project includes a dedicated evaluation setup in the `/eval` folder.

### Contents

```text
/eval
  ├── run_eval.py
  └── /test_prompts
        └── test_prompts.jsonl
```

- `test_prompts.jsonl` contains **10 curated test questions**
  - Mix of valid, invalid, and adversarial prompts

- `run_eval.py`:
  - Executes the full graph for each prompt
  - Verifies correctness, grounding, and failure handling

### Run Evaluation

```bash
uv run python -m eval.run_eval
```

---

## Streamlit User Interface

A lightweight UI is provided for interactive usage.

**Location:**

```text
/app/streamlit_app.py
```

### Run the UI

```bash
uv run streamlit run app/streamlit_app.py
```

## Environment Setup

### Required Environment Variable

The application requires an OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

This should be set in an `.env` file

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
