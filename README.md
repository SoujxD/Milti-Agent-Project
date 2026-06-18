# Multi-Agent Business Analytics

A LangGraph-orchestrated, RAG-powered analyst that turns an ecommerce CSV into
structured business answers, benchmarks itself, and exports a stakeholder
PowerPoint deck. Built on **LangChain**, **LangGraph**, **ChromaDB**, **Ragas**,
and **Pydantic**, with a Streamlit UI.

Runs fully **offline with no API key** via a deterministic mock fallback, so
demos and grading work without any paid models.

## What's inside

| Module | What it does |
|---|---|
| `utils/lc_retriever.py` | Persistent ChromaDB vector store using HuggingFace MiniLM embeddings, with a TF-IDF `BaseRetriever` fallback. |
| `agents/analyst_agent.py` | Two analyst paths: legacy `AnalystRAGAgent` (used by the factorial eval) and `run_analyst_lc` returning a strict Pydantic `AnalystAnswer`. |
| `agents/graph.py` | LangGraph supervisor wiring `START -> supervisor -> {analyst, presenter, END}` with LLM-based or rule-based routing. |
| `agents/presentation_agent.py` | Auto-detects schema and builds a 7-slide `.pptx` with charts and speaker notes. |
| `evaluation/evaluator.py` | Factorial benchmark across models x prompt styles x RAG on/off with 10+ scoring metrics. |
| `evaluation/ragas_eval.py` | Ragas RAG quality grader: Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall. |
| `ui/app.py` | Streamlit UI: Overview / EDA / Analyst / Presentation / Evaluation tabs. |
| `api/server.py` | FastAPI server exposing the analyst, summary, and presentation endpoints. |

## Stack

- **LLM:** `ChatOpenAI` (gpt-4o-mini) via LangChain, or OpenRouter; deterministic mock when no key is set.
- **Vector store:** ChromaDB persistent index, `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **Multi-agent orchestration:** LangGraph supervisor pattern.
- **Structured outputs:** Pydantic + `with_structured_output`.
- **Evaluation:** Ragas + a custom factorial benchmark.

## Quick start

```bash
git clone https://github.com/SoujxD/Milti-Agent-Project.git
cd Milti-Agent-Project
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for real LLM calls (everything still works without these):

```bash
# bash / zsh
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=or-...

# PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:OPENROUTER_API_KEY="or-..."
```

## Run it

| Command | What it does |
|---|---|
| `python main.py demo --question "..."` | Single legacy analyst answer. |
| `python main.py graph --question "..."` | LangGraph supervisor (analyst, plus presenter if the question mentions a deck). |
| `python main.py evaluate --limit 20 --no-judge` | Factorial benchmark grid (models x prompts x RAG on/off). |
| `python main.py ragas --limit 20` | Ragas RAG evaluation (skips gracefully without a key). |
| `python main.py presentation` | One-click PowerPoint deck from the dataset. |
| `streamlit run ui/app.py` | Full UI at `http://localhost:8501`. |
| `uvicorn api.server:app --reload --port 8000` | FastAPI backend. |

## How the LangGraph supervisor decides

Without an LLM key, the supervisor uses a deterministic rule:

- No analysis yet -> `analyst`.
- Question mentions `deck` / `slides` / `presentation` / `powerpoint` / `ppt` and no deck yet -> `presenter`.
- Otherwise -> `FINISH`.

With `OPENAI_API_KEY` set, the supervisor instead uses
`ChatOpenAI(...).with_structured_output(RouterDecision)` to route between
`analyst`, `presenter`, and `FINISH`.

## Dataset

The bundled `data/dataset.csv` follows the **UCI Online Shoppers Intention**
schema (page activity, durations, bounce/exit rates, page values, visitor type,
traffic source, binary revenue label). The retriever turns each row into a
Document and adds one summary Document with dataset-level aggregates. The
first run builds and persists a Chroma index at `data/chroma_db/` (gitignored);
subsequent runs load it from disk in milliseconds.

The 100-question bank at `data/evaluation_questions.json` includes
`question`, `category`, `expected_variables`, `expected_keywords`,
`ground_truth`, and `numeric_answer` for scoring.

## Evaluation metrics

`python main.py evaluate` scores each analyst run on:

- `keyword_score`, `recommendation_score`, `completeness_score`
- `groundedness_score`, `business_specificity_score`, `retrieval_usefulness_score`
- `json_validity_score`, `unique_insight_ratio`, `avg_response_length`, `insight_count`
- Optional LLM-as-judge ratings (`usefulness`, `clarity`, `correctness`) with a heuristic fallback when no key is set.

`python main.py ragas` scores:

- **Faithfulness**, **ResponseRelevancy**, **LLMContextPrecisionWithoutReference**, **LLMContextRecall**.
  Requires an LLM judge; without `OPENAI_API_KEY` it writes a `{"status": "skipped"}` placeholder.

## Mock vs real

| Component | With API key | Without |
|---|---|---|
| Old analyst (`answer_question`) | Calls OpenRouter | Deterministic JSON from prompt+model hash |
| New analyst (`run_analyst_lc`) | `ChatOpenAI` with structured output | Mock JSON coerced into `AnalystAnswer` |
| Supervisor routing | LLM picks the next agent | Rule-based |
| LLM-as-judge (legacy eval) | Real judge model | Rubric-based heuristic |
| Ragas | Real Faithfulness/Relevancy/Context scores | Skips cleanly |
| Embeddings | MiniLM (local) | MiniLM (local) |

## Deployment

- **Streamlit UI**: `streamlit run ui/app.py` or deploy to Streamlit Community Cloud.
- **FastAPI backend**: Dockerfile + `fly.toml` included. Deploy with `flyctl deploy`. Configure the GitHub Pages frontend by setting `API_BASE_URL` in `docs/config.js`.
- **GitHub Pages**: serve the static site from `docs/` (Settings -> Pages -> Deploy from branch, select `main` and `/docs`).

## Folder layout

```
agents/        analyst, presenter, LangGraph supervisor
api/           FastAPI server
data/          dataset.csv, evaluation_questions.json, sample generator
docs/          static GitHub Pages site
evaluation/    factorial benchmark + Ragas pipeline
ui/            Streamlit app
utils/         retrievers, LLM client, parsers, dataset adapter
main.py        CLI entrypoint
```

## Design notes

- The mock fallback is preserved end-to-end: ChatOpenAI -> mock; LLM-judge -> heuristic; Chroma -> TF-IDF; Ragas -> skip.
- The TF-IDF fallback is wrapped behind a LangChain `BaseRetriever` so the rest of the code stays retriever-agnostic.
- The LangGraph presenter currently triggers a dataset-driven deck and does not yet thread `analyst_output` into slide content (a planned refinement).
- Real Ragas scores require a paid LLM judge; the no-key run is a clean skip.

## Versions

Built against:

- `langchain 1.3+`, `langchain-core 1.4+`, `langgraph 1.2+`, `langgraph-supervisor 0.0.31`
- `chromadb 1.5+`, `langchain-chroma 1.1+`
- `ragas 0.4+`, `datasets 4.8+`
- `langfuse 4.6+`, `pydantic 2.7+`
- Python 3.11+
