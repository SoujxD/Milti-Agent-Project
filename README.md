# Multi-Agent Business Analytics System

This project is a production-style Generative AI course project that combines two cooperating agents with a rigorous evaluation loop:

- A **Data Analyst RAG Agent** answers business questions over an e-commerce dataset.
- A **Presentation Generator Agent** turns dataset insights into charts and a stakeholder-ready PowerPoint deck.
- A **Streamlit web app** exposes the workflow in an interactive executive-style demo.
- A **GitHub Pages static site** gives you a public-facing link for project submission and reporting.
- A **multi-model, multi-prompt evaluation pipeline** compares prompt strategies, model choices, and RAG versus no-RAG behavior.

The project is designed to run end to end even without paid APIs. If an `OPENROUTER_API_KEY` is available, the analyst agent will call OpenRouter. Otherwise it falls back to a deterministic mock model so the system remains fully runnable for demos and grading.

## Folder Structure

```text
project/
├── agents/
│   ├── __init__.py
│   ├── analyst_agent.py
│   └── presentation_agent.py
├── api/
│   ├── __init__.py
│   └── server.py
├── data/
│   ├── dataset.csv
│   ├── evaluation_questions.json
│   └── generate_sample_data.py
├── docs/
│   ├── index.html
│   ├── analyst.html
│   ├── evaluation.html
│   ├── presentation.html
│   ├── app.js
│   ├── config.js
│   └── styles.css
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py
├── outputs/
│   └── ...
├── ui/
│   └── app.py
├── utils/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── parser.py
│   └── retriever.py
├── Dockerfile
├── fly.toml
├── main.py
├── README.md
├── requirements.txt
└── requirements.fly.txt
```

## System Architecture

The architecture is intentionally modular:

1. `utils/retriever.py` loads the e-commerce dataset and builds a retriever.
2. `agents/analyst_agent.py` retrieves relevant evidence, builds a prompt, and requests a JSON answer from the selected model.
3. `evaluation/evaluator.py` runs the analyst agent across 4 models, 4 prompt styles, and 100 questions, with both RAG on and off.
4. `evaluation/metrics.py` computes quantitative metrics.
5. `agents/presentation_agent.py` creates dataset-focused charts and exports a 7-slide PowerPoint deck.
6. `ui/app.py` exposes all major functionality in a Streamlit web app.
7. `docs/index.html` serves as the static GitHub Pages site.

Architecture flow:

```text
Business question
  -> Retriever (FAISS if available, TF-IDF fallback)
  -> Analyst Agent
  -> Structured JSON output
  -> Evaluation Pipeline
  -> Aggregated metrics + top examples
  -> Presentation Generator Agent
  -> Streamlit web interface / downloadable PPT
```

## Dataset

The sample dataset follows the **Online Shoppers Intention** schema and includes:

- Administrative and informational browsing activity
- Product-related activity and durations
- Bounce and exit rates
- Page values and seasonal features
- Visitor type, browser, region, and traffic source
- Binary revenue label

The included `data/generate_sample_data.py` script creates:

- `data/dataset.csv`
- `data/evaluation_questions.json`

The evaluation file contains 100 business questions with:

- `question`
- `category`
- `expected_variables`
- `expected_type`
- `ground_truth`
- `numeric_answer`
- `expected_keywords`

## Metrics

The evaluation pipeline computes four metrics:

- `keyword_score`: fraction of expected keywords covered by the answer
- `recommendation_score`: measures whether recommendations are present and actionable
- `completeness_score`: checks whether all required JSON fields are populated
- `groundedness_score`: estimates overlap between the generated answer and retrieved context

An `overall_score` is also computed as the simple average of the four metrics.

## Installation

Use Python 3.10+.

```bash
cd /Users/shubhangimittal/Desktop/ISE547/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want live model calls:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Generate Data

If you want to regenerate the sample data:

```bash
python data/generate_sample_data.py
```

## Run the Analyst Agent Demo

```bash
python main.py demo \
  --question "Which traffic sources and visitor segments should we prioritize?" \
  --model meta-llama/llama-3.1-8b-instruct \
  --prompt-style structured_json
```

Disable RAG for comparison:

```bash
python main.py demo \
  --question "How can we improve conversion?" \
  --model mistralai/mistral-7b-instruct \
  --prompt-style executive \
  --no-rag
```

## Run Experiments

Full evaluation:

```bash
python main.py evaluate
```

Quick smoke test:

```bash
python main.py evaluate --limit 20
```

Outputs written to `outputs/`:

- `results.csv`
- `model_comparison.csv`
- `prompt_comparison.csv`
- `rag_comparison.csv`
- `category_comparison.csv`

## Generate the Presentation

```bash
python main.py presentation
```

The presentation agent creates:

- `outputs/presentation.pptx`
- chart images under `outputs/charts/`
- mirrored public assets under `docs/assets/`

The generated deck is intentionally focused on the dataset and business insights:

1. Title
2. Dataset Snapshot
3. Target Customers
4. Behavioral Patterns
5. Recommendations
6. Key Findings
7. Conclusion

Each slide includes:

- a title
- up to 5 bullets
- speaker notes

## Launch the Streamlit Website

```bash
streamlit run ui/app.py
```

The UI contains four tabs:

1. **Overview**
2. **Analyst Agent Demo**
3. **Evaluation Dashboard**
4. **Presentation Generator**

The site supports model and prompt selection, RAG toggling, evaluation visualization, executive text responses, slide preview, and PPT download.

## Public Website via GitHub Pages

Use the `docs/` folder to publish a public website:

1. Push the repository to GitHub.
2. Open **Settings -> Pages** in the GitHub repository.
3. Choose **Deploy from a branch**.
4. Select your main branch and the `/docs` folder.
5. Save the settings to receive a public URL.

Important deployment note:

- GitHub Pages can host the static project site in `docs/`.
- GitHub Pages cannot run the Python agents directly.
- This repository now includes a FastAPI backend in `api/server.py` so the public HTML pages can call live analyst and presentation endpoints.
- Use `docs/` on GitHub Pages for the frontend and deploy the FastAPI backend separately on Render, Railway, or another Python host.

## Public App Architecture

Recommended production split:

1. **Frontend**: GitHub Pages serving `docs/`
2. **Backend**: FastAPI serving `api/server.py`

Available backend endpoints:

- `GET /health`
- `POST /api/summary`
- `POST /api/analyst`
- `POST /api/presentation`
- `GET /api/downloads/{filename}`

The HTML frontend will use the backend automatically when `docs/config.js` contains a non-empty `API_BASE_URL`.

Example:

```js
window.APP_CONFIG = {
  API_BASE_URL: "https://your-backend-service.onrender.com"
};
```

## Deploy the Backend on Fly.io

This repo includes:

- [Dockerfile](/Users/shubhangimittal/Desktop/ISE547/project/Dockerfile)
- [fly.toml](/Users/shubhangimittal/Desktop/ISE547/project/fly.toml)
- [requirements.fly.txt](/Users/shubhangimittal/Desktop/ISE547/project/requirements.fly.txt)

Basic steps:

1. Push the repository to GitHub.
2. Authenticate with Fly.io using `flyctl auth login`.
3. Create the app if needed: `flyctl apps create <app-name>`.
4. Set required secrets such as:
   - `flyctl secrets set OPENROUTER_API_KEY=... -a <app-name>`
   - `flyctl secrets set ALLOWED_ORIGINS=https://your-github-pages-domain -a <app-name>`
5. Deploy with `flyctl deploy -c fly.toml -a <app-name>`.
6. Copy the deployed backend URL into `docs/config.js`.
7. Push again and redeploy GitHub Pages.

## Local Backend Run

You can also run the API locally:

```bash
uvicorn api.server:app --reload --port 8000
```

Then set:

```js
window.APP_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000"
};
```

## Prompt Styles

The analyst agent supports four prompt styles:

- `basic`
- `structured_json`
- `executive`
- `evidence_constrained`

## Models

By default the system evaluates these four inexpensive-compatible model identifiers:

- `meta-llama/llama-3.1-8b-instruct`
- `mistralai/mistral-7b-instruct`
- `google/gemma-2-9b-it`
- `qwen/qwen-2.5-7b-instruct`

These can be replaced with any OpenRouter-supported models.

## Error Handling and Design Notes

- The retriever uses a safe TF-IDF fallback by default and only enables sentence-transformer embeddings when `ENABLE_SENTENCE_TRANSFORMERS=true`.
- The LLM client falls back to a deterministic mock JSON generator if API access fails.
- JSON parsing is tolerant of fenced code blocks and malformed outputs.
- The Streamlit UI warns the user when evaluation artifacts are missing.

## Future Improvements

- Replace heuristic metrics with LLM-as-a-judge and human evaluation.
- Add persistent vector storage instead of rebuilding retrieval in memory.
- Support direct use of the original Kaggle dataset instead of the bundled sample data.
- Add authentication and deployment configuration for a public demo site.
- Extend the presentation agent with branded templates and richer notes formatting.
