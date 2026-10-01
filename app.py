import asyncio
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from generate_profile_pdf import build_pdf

APP_ROOT = Path(__file__).resolve().parent
PHASE0_DIR = APP_ROOT / "phase0_proof_of_concept"
if str(PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE0_DIR))

from phase0_proof_of_concept.analyze_learner import build_larger_context
from phase0_proof_of_concept.extractor import ChatGPTUserMessageExtractor, ParseError
from phase0_proof_of_concept.profile_builder import LLMProvider, ProfileBuilderError, build_profile

MAX_UPLOAD_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ARCHIVE_JSON_BYTES = 600 * 1024 * 1024
STAGED_UPLOAD_TTL_SECONDS = 30 * 60
STAGED_UPLOAD_DIR = Path(tempfile.gettempdir()) / "learner-profile-staged-uploads"
logger = logging.getLogger(__name__)
_staged_uploads: dict[str, tuple[Path, str, float]] = {}
_staged_uploads_lock = threading.Lock()


def _remove_expired_staged_uploads() -> None:
    now = time.monotonic()
    with _staged_uploads_lock:
        expired = [
            upload_id
            for upload_id, (_, _, expires_at) in _staged_uploads.items()
            if now >= expires_at
        ]
        paths = [_staged_uploads.pop(upload_id)[0] for upload_id in expired]
    for path in paths:
        path.unlink(missing_ok=True)


def _take_staged_upload(upload_id: str) -> tuple[Path, str] | None:
    _remove_expired_staged_uploads()
    with _staged_uploads_lock:
        staged = _staged_uploads.pop(upload_id, None)
    if staged is None:
        return None
    path, filename, _ = staged
    return path, filename


async def _staged_upload_cleanup_loop() -> None:
    while True:
        _remove_expired_staged_uploads()
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    shutil.rmtree(STAGED_UPLOAD_DIR, ignore_errors=True)
    STAGED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_task = asyncio.create_task(_staged_upload_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Learner AI Tutor", lifespan=lifespan)


class ProfileUploadError(ValueError):
    """Raised when an uploaded file is not a supported learner profile."""


def _messages_from_upload(source_path: Path) -> list[dict[str, Any]]:
    """Read either extracted user messages or a ChatGPT conversation export."""
    if source_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source_path) as archive:
                conversations = [
                    item for item in archive.infolist()
                    if not item.is_dir()
                    and item.filename.lower().endswith(".json")
                    and "conversations" in Path(item.filename).stem.lower()
                ]
                if not conversations:
                    raise ProfileUploadError(
                        "The ZIP does not contain a ChatGPT conversations JSON export."
                    )
                expanded_bytes = sum(item.file_size for item in conversations)
                if expanded_bytes > MAX_ARCHIVE_JSON_BYTES:
                    raise ProfileUploadError(
                        "The conversations inside this ZIP expand beyond the 600 MB processing limit."
                    )
        except zipfile.BadZipFile as exc:
            raise ProfileUploadError("This ZIP file is invalid or damaged.") from exc

    if source_path.suffix.lower() == ".json":
        try:
            data = json.loads(source_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProfileUploadError("This file is not valid UTF-8 JSON.") from exc

        if isinstance(data, list) and data and all(
            isinstance(item, dict)
            and isinstance(item.get("content"), str)
            and ("conversation_id" in item or "message_id" in item)
            for item in data
        ):
            return data

    try:
        extracted = ChatGPTUserMessageExtractor(source_path).extract_all_user_messages()
    except (OSError, ValueError, ParseError) as exc:
        raise ProfileUploadError(
            "Could not read this conversation export. Upload a ChatGPT export ZIP, "
            "a conversations JSON file, or an extracted user_messages JSON file. "
            f"Details: {exc}"
        ) from exc

    messages = [asdict(message) for message in extracted]
    if not messages:
        raise ProfileUploadError("No user-authored text messages were found in this export.")
    return messages


def generate_profile_from_conversation_upload(
    uploaded_bytes: bytes,
    filename: str,
    api_key: str | None = None,
    llm: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, int]]:
    """Run extraction, compact analysis, LLM profiling, and PDF rendering in temporary storage."""
    extension = Path(filename).suffix.lower()
    if extension not in {".zip", ".json"}:
        raise ProfileUploadError("Upload a ChatGPT export .zip or conversations .json file.")
    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        raise ProfileUploadError(f"This upload exceeds the {MAX_UPLOAD_MB} MB limit.")

    def report(message: str) -> None:
        if progress:
            progress(message)

    with tempfile.TemporaryDirectory(prefix="learner-profile-run-") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / f"conversation_export{extension}"
        messages_path = temp_path / "user_messages.json"
        context_path = temp_path / "rich_learner_context.json"
        pdf_path = temp_path / "learner_activity_profile.pdf"
        source_path.write_bytes(uploaded_bytes)

        report("Extracting your messages…")
        messages = _messages_from_upload(source_path)
        messages_path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")

        report("Selecting and organizing evidence…")
        context = build_larger_context(messages_path, context_path)
        if not context or not context["dataset_summary"]["selected_conversations"]:
            raise ProfileUploadError(
                "The export was read, but no topic-specific learning evidence was selected."
            )

        report("Building the evidence-cited profile with Groq…")
        provider = llm or LLMProvider(api_key=api_key)
        profile = build_profile(context_path, provider)

        report("Rendering the PDF…")
        build_pdf(profile, pdf_path)
        stats = {
            "messages": len(messages),
            "conversations": context["dataset_summary"]["total_conversations"],
            "selected_conversations": context["dataset_summary"]["selected_conversations"],
            "evidence_items": context["dataset_summary"]["selected_evidence_items"],
        }
        return profile, pdf_path.read_bytes(), stats


def generate_pdf_from_bytes(uploaded_bytes: bytes) -> tuple[dict[str, Any], bytes]:
    """Validate an uploaded profile and render its PDF using temporary local files."""
    try:
        profile = json.loads(uploaded_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileUploadError("This file is not valid UTF-8 JSON.") from exc

    if not isinstance(profile, dict):
        raise ProfileUploadError("The uploaded JSON must be an object.")

    required = ("topic_activity", "observed_patterns", "explicit_preferences", "evidence")
    missing = [key for key in required if not isinstance(profile.get(key), list)]
    if missing:
        raise ProfileUploadError(
            "This is not an evidence-cited learner profile. Missing sections: "
            + ", ".join(missing)
            + ". Upload the learner_profile.json produced by the current profile builder."
        )

    evidence_ids = {
        item.get("evidence_id")
        for item in profile["evidence"]
        if isinstance(item, dict) and item.get("evidence_id")
    }
    for section in ("observed_patterns", "explicit_preferences"):
        for claim in profile[section]:
            if not isinstance(claim, dict) or not isinstance(claim.get("evidence_ids"), list):
                raise ProfileUploadError(f"Every item in '{section}' must include evidence IDs.")
            if not set(claim["evidence_ids"]) <= evidence_ids:
                raise ProfileUploadError(f"A claim in '{section}' references missing evidence.")

    with tempfile.TemporaryDirectory(prefix="learner-profile-") as temp_dir:
        input_path = Path(temp_dir) / "profile.json"
        output_path = Path(temp_dir) / "profile.pdf"
        input_path.write_bytes(uploaded_bytes)
        try:
            build_pdf(profile, output_path)
        except (OSError, ValueError) as exc:
            raise ProfileUploadError(str(exc)) from exc
        return profile, output_path.read_bytes()


def _validate_preview(profile: dict[str, Any]) -> None:
    if not isinstance(profile.get("topic_activity"), list):
        raise ProfileUploadError("This profile does not contain a topic_activity list.")
    if not isinstance(profile.get("observed_patterns"), list):
        raise ProfileUploadError("This profile does not contain an observed_patterns list.")
    if not isinstance(profile.get("explicit_preferences"), list):
        raise ProfileUploadError("This profile does not contain an explicit_preferences list.")
    if not isinstance(profile.get("evidence"), list):
        raise ProfileUploadError("This profile does not contain an evidence list.")


@app.get("/")
async def read_root() -> FileResponse:
    return FileResponse(str(APP_ROOT / "templates" / "index.html"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_export(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Upload a ChatGPT export .zip file.")

    _remove_expired_staged_uploads()
    upload_id = uuid.uuid4().hex
    staged_path = STAGED_UPLOAD_DIR / f"{upload_id}.zip"
    size = 0

    try:
        with staged_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"This upload exceeds the {MAX_UPLOAD_MB} MB limit.",
                    )
                destination.write(chunk)
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise

    if size == 0:
        staged_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with _staged_uploads_lock:
        _staged_uploads[upload_id] = (
            staged_path,
            Path(file.filename).name,
            time.monotonic() + STAGED_UPLOAD_TTL_SECONDS,
        )
    return {"upload_id": upload_id, "filename": Path(file.filename).name, "size": size}


@app.post("/analyze")
async def analyze_export(
    file: UploadFile | None = File(default=None),
    upload_id: str | None = Form(default=None),
    api_key: str | None = Form(default=None),
):
    if upload_id:
        staged = _take_staged_upload(upload_id)
        if staged is None:
            raise HTTPException(
                status_code=410,
                detail="This upload expired or was already used. Please upload the file again.",
            )
        staged_path, filename = staged
        try:
            contents = staged_path.read_bytes()
        finally:
            staged_path.unlink(missing_ok=True)
    elif file and file.filename:
        filename = file.filename
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    else:
        raise HTTPException(status_code=400, detail="No uploaded file was provided.")

    try:
        _, pdf_bytes, _ = generate_profile_from_conversation_upload(
            uploaded_bytes=contents,
            filename=filename,
            api_key=api_key,
        )
    except ProfileUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProfileBuilderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected runtime failures
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail="Unexpected server error while generating the profile PDF.") from exc

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="learner_profile.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)