"""FastAPI backend for the GitHub Pages frontend."""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from agents.analyst_agent import AnalystRAGAgent
from agents.eda_agent import EDAAgent
from agents.presentation_agent import PresentationGeneratorAgent
from utils.dataset_adapter import read_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
API_UPLOAD_DIR = OUTPUT_DIR / "api_uploads"
API_PRESENTATION_DIR = OUTPUT_DIR / "api_presentations"
API_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
API_PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)


def _allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "*")
    if configured.strip() == "*":
        return ["*"]
    return [item.strip() for item in configured.split(",") if item.strip()]


app = FastAPI(title="ISE547 Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "dataset.csv").suffix or ".csv"
    file_id = uuid.uuid4().hex
    destination = API_UPLOAD_DIR / f"{file_id}{suffix}"
    content = upload.file.read()
    destination.write_bytes(content)
    return destination


def _validate_csv(upload_path: Path) -> pd.DataFrame:
    try:
        return read_dataset(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse dataset file: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/summary")
async def summarize_dataset(file: UploadFile = File(...)) -> dict:
    upload_path = _save_upload(file)
    dataset = await run_in_threadpool(_validate_csv, upload_path)
    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    summary = await run_in_threadpool(presenter.summarize_dataset, upload_path)
    return {
        "dataset_name": file.filename,
        "rows": summary["rows"],
        "columns": summary["columns"],
        "analysis_rows": summary["analysis_rows"],
        "analysis_grain": summary["analysis_grain"],
        "source_format": summary["source_format"],
        "target_rate": summary["target_rate"],
        "top_time": summary["top_time"],
        "top_customer": summary["top_customer"],
        "top_channel": summary["top_channel"],
        "schema": summary["schema"],
        "preview": dataset.head(12).fillna("").to_dict(orient="records"),
    }


@app.post("/api/analyst")
async def analyst_answer(
    file: UploadFile = File(...),
    question: str = Form(...),
    model: str = Form("meta-llama/llama-3.1-8b-instruct"),
    prompt_style: str = Form("structured_json"),
    rag_enabled: bool = Form(True),
    top_k: int = Form(5),
) -> dict:
    upload_path = _save_upload(file)
    await run_in_threadpool(_validate_csv, upload_path)
    agent = AnalystRAGAgent(dataset_path=upload_path)
    result = await run_in_threadpool(
        agent.answer_question,
        question,
        model,
        prompt_style,
        rag_enabled,
        top_k,
    )
    return {
        "question": result.question,
        "model": result.model,
        "prompt_style": result.prompt_style,
        "rag_enabled": result.rag_enabled,
        "parsed_response": result.parsed_response,
        "raw_response": result.raw_response,
        "retrieved_context": result.retrieved_context,
    }


@app.post("/api/eda")
async def profile_dataset(file: UploadFile = File(...)) -> dict:
    """Run deterministic EDA profiling for the public HTML UI."""
    upload_path = _save_upload(file)
    await run_in_threadpool(_validate_csv, upload_path)
    eda_agent = EDAAgent(output_dir=OUTPUT_DIR)
    report_id = uuid.uuid4().hex[:12]
    report = await run_in_threadpool(
        eda_agent.analyze_dataset,
        upload_path,
        True,
        f"api_{report_id}",
        None,
    )
    return {
        "dataset_name": file.filename,
        "profile": report["profile"],
        "quality_checks": report["quality_checks"],
        "chart_manifest": [
            {
                **chart,
                "chart_url": f"/api/downloads/{chart['filename']}",
            }
            for chart in report["chart_manifest"]
        ],
        "suggested_questions": report["suggested_questions"],
        "handoff_summary": report["handoff_summary"],
        "retrieval_chunks": report["retrieval_chunks"],
        "key_findings": report["key_findings"],
        "preview": report["preview"],
    }


@app.post("/api/presentation")
async def generate_presentation(file: UploadFile = File(...)) -> dict:
    upload_path = _save_upload(file)
    await run_in_threadpool(_validate_csv, upload_path)
    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    output_filename = f"{uuid.uuid4().hex}_presentation.pptx"
    output_path = API_PRESENTATION_DIR / output_filename
    ppt_path, slides, chart_paths = await run_in_threadpool(presenter.create_presentation, upload_path, output_path)
    return {
        "download_url": f"/api/downloads/{ppt_path.name}",
        "slides": [
            {
                "title": slide.title,
                "bullets": slide.bullets,
                "speaker_notes": slide.speaker_notes,
                "chart_key": slide.chart_key,
                "chart_url": f"/api/downloads/{chart_paths[slide.chart_key].name}" if slide.chart_key and slide.chart_key in chart_paths else None,
            }
            for slide in slides
        ],
    }


@app.get("/api/downloads/{filename}")
def download_artifact(filename: str):
    for directory in [API_PRESENTATION_DIR, OUTPUT_DIR / "charts"]:
        candidate = directory / filename
        if candidate.exists():
            media_type = None
            if candidate.suffix.lower() == ".pptx":
                media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif candidate.suffix.lower() == ".png":
                media_type = "image/png"
            return FileResponse(candidate, media_type=media_type, filename=candidate.name)
    raise HTTPException(status_code=404, detail="Artifact not found")
