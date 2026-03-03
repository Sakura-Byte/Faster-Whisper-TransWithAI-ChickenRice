from __future__ import annotations

import argparse
import logging
import os
import secrets
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .infer import Inference

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FORMAT = "lrc"
DEFAULT_LANGUAGE = "zh-CN"
SUPPORTED_FORMATS = {"lrc", "srt", "vtt"}
ALL_AUDIO_SUFFIXES = "wav,flac,mp3,m4a,aac,ogg,wma,mkv,avi,mov,webm,flv,wmv,mp4"


class TranscribeRequest(BaseModel):
    download_url: str = Field(..., description="Source audio URL")
    download_headers: dict[str, str] = Field(default_factory=dict)
    language: str = DEFAULT_LANGUAGE
    output_format: str = DEFAULT_OUTPUT_FORMAT
    device: str = "cuda"
    compute_type: str = "auto"
    enable_batching: bool = True
    batch_size: int | None = None
    max_batch_size: int = 8


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    stage: str = "queued"
    progress_pct: float = 0.0
    message: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress_pct": round(float(self.progress_pct), 2),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return {**payload, "data": payload}


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def create(self) -> JobState:
        job = JobState(job_id=uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress_pct: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress_pct is not None:
                job.progress_pct = max(0.0, min(100.0, progress_pct))
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = datetime.now(timezone.utc)


API_TOKEN = os.getenv("CHICKENRICE_API_TOKEN", "").strip()
MODEL_PATH = os.getenv("CHICKENRICE_MODEL_PATH", "models").strip() or "models"
GENERATION_CONFIG = os.getenv("CHICKENRICE_GENERATION_CONFIG", "generation_config.json5").strip() or "generation_config.json5"
JOB_WORKERS = max(1, int(os.getenv("CHICKENRICE_JOB_WORKERS", "1")))
DOWNLOAD_TIMEOUT_SECONDS = max(10, int(os.getenv("CHICKENRICE_DOWNLOAD_TIMEOUT_SECONDS", "600")))

JOB_STORE = JobStore()
EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="chickenrice-job")

app = FastAPI(title="ChickenRice WebAPI", version="1.0.0")


def _auth_guard(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return

    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="invalid authorization header")

    if not secrets.compare_digest(token.strip(), API_TOKEN):
        raise HTTPException(status_code=403, detail="invalid token")


def _normalize_format(fmt: str | None) -> str:
    value = (fmt or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if value not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported output_format: {value}")
    return value


def _normalize_language(language: str | None) -> tuple[str, str]:
    value = (language or DEFAULT_LANGUAGE).strip().lower()

    # ChickenRice is tuned for Japanese audio.
    if value in {"zh", "zh-cn", "zh_hans", "zh-tw", "zh_hant", "cn"}:
        return "ja", "translate"
    if value in {"ja", "ja-jp", "jp"}:
        return "ja", "transcribe"
    if value in {"en", "en-us", "en-gb"}:
        return "en", "transcribe"

    # Fallback: best-effort language code extraction.
    if "-" in value:
        value = value.split("-", 1)[0]
    return value, "transcribe"


def _guess_extension(download_url: str) -> str:
    path = urlparse(download_url).path
    ext = Path(path).suffix.lower()
    if ext:
        return ext
    return ".audio"


def _download_to_file(download_url: str, download_headers: dict[str, str], directory: Path) -> Path:
    extension = _guess_extension(download_url)
    target = directory / f"source{extension}"

    with requests.get(download_url, headers=download_headers or {}, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
        resp.raise_for_status()
        with target.open("wb") as file_obj:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    return target


def _build_inference_args(
    *,
    output_dir: Path,
    device: str,
    compute_type: str,
    output_format: str,
    enable_batching: bool,
    batch_size: int | None,
    max_batch_size: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_name_or_path=MODEL_PATH,
        device=device,
        compute_type=compute_type,
        overwrite=True,
        audio_suffixes=ALL_AUDIO_SUFFIXES,
        sub_formats=output_format,
        output_dir=str(output_dir),
        generation_config=GENERATION_CONFIG,
        log_level=os.getenv("CHICKENRICE_LOG_LEVEL", "INFO"),
        merge_segments=None,
        merge_max_gap_ms=None,
        merge_max_duration_ms=None,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_min_silence_duration_ms=None,
        vad_speech_pad_ms=None,
        console=False,
        enable_batching=enable_batching,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        base_dirs=[],
    )


def _run_inference(req: TranscribeRequest, progress_cb) -> dict[str, Any]:
    output_format = _normalize_format(req.output_format)
    model_language, model_task = _normalize_language(req.language)

    with tempfile.TemporaryDirectory(prefix="chickenrice-api-") as tmp:
        tmp_dir = Path(tmp)
        output_dir = tmp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_cb("downloading", 10, "downloading source")
        source_path = _download_to_file(req.download_url, req.download_headers, tmp_dir)

        args = _build_inference_args(
            output_dir=output_dir,
            device=req.device,
            compute_type=req.compute_type,
            output_format=output_format,
            enable_batching=req.enable_batching,
            batch_size=req.batch_size,
            max_batch_size=req.max_batch_size,
        )

        progress_cb("transcribing", 35, "transcribing")
        start = time.perf_counter()

        inference = Inference(args)
        # Override generation config dynamically for API requests.
        inference.generation_config["language"] = model_language
        inference.generation_config["task"] = model_task
        inference.generates([str(source_path)])

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        progress_cb("formatting", 90, "formatting subtitle")

        expected_output = output_dir / f"{source_path.stem}.{output_format}"
        if expected_output.exists():
            output_path = expected_output
        else:
            candidates = list(output_dir.rglob(f"*.{output_format}"))
            if not candidates:
                raise RuntimeError("subtitle output not found")
            output_path = candidates[0]

        content = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            raise RuntimeError("empty subtitle output")

        progress_cb("saving", 96, "finalizing result")
        return {
            "content": content,
            "format": output_format,
            "language": req.language,
            "duration_ms": elapsed_ms,
            "device": req.device,
            "compute_type": req.compute_type,
        }


def _run_job(job_id: str, req: TranscribeRequest) -> None:
    def update(stage: str, progress_pct: float, message: str) -> None:
        JOB_STORE.update(
            job_id,
            status="running",
            stage=stage,
            progress_pct=progress_pct,
            message=message,
        )

    try:
        update("queued", 2, "queued")
        result = _run_inference(req, update)
        JOB_STORE.update(
            job_id,
            status="done",
            stage="done",
            progress_pct=100,
            message="completed",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        JOB_STORE.update(
            job_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message=str(exc),
            error=str(exc),
        )


@app.get("/healthz", dependencies=[Depends(_auth_guard)])
def healthz() -> dict[str, Any]:
    payload = {
        "status": "ok",
        "stage": "ready",
        "progress_pct": 100,
        "message": "healthy",
        "time": datetime.now(timezone.utc).isoformat(),
    }
    return {**payload, "data": payload}


@app.post("/v1/transcribe", dependencies=[Depends(_auth_guard)])
def transcribe_sync(request: TranscribeRequest) -> dict[str, Any]:
    stage_holder = {"stage": "queued", "progress": 0.0, "message": "queued"}

    def update(stage: str, progress_pct: float, message: str) -> None:
        stage_holder["stage"] = stage
        stage_holder["progress"] = progress_pct
        stage_holder["message"] = message

    result = _run_inference(request, update)
    payload = {
        "status": "done",
        "stage": "done",
        "progress_pct": 100,
        "message": "completed",
        "result": result,
    }
    return {**payload, "data": payload}


@app.post("/v1/jobs/transcribe", dependencies=[Depends(_auth_guard)])
def create_transcribe_job(request: TranscribeRequest) -> dict[str, Any]:
    job = JOB_STORE.create()
    EXECUTOR.submit(_run_job, job.job_id, request)

    payload = {
        "job_id": job.job_id,
        "status": "queued",
        "stage": "queued",
        "progress_pct": 0,
        "message": "queued",
    }
    return {**payload, "data": payload}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(_auth_guard)])
def get_job(job_id: str) -> dict[str, Any]:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_payload()


def main() -> None:
    log_level = os.getenv("CHICKENRICE_API_LOG_LEVEL", "info").lower()
    host = os.getenv("CHICKENRICE_API_HOST", "0.0.0.0")
    port = int(os.getenv("CHICKENRICE_API_PORT", "8000"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "starting webapi host=%s port=%s model_path=%s workers=%s",
        host,
        port,
        MODEL_PATH,
        JOB_WORKERS,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
