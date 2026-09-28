"""
Reads user_messages.json (output of extractor.py) and produces a structured,
confidence-scored learner profile by batching messages through an LLM.

Pipeline:
    user_messages.json -> batch into groups -> LLM call per batch
    -> validate/parse each batch's JSON output -> merge all batches
    -> final learner_profile.json

Design notes:
- The LLM call is isolated behind LLMProvider so the provider can be swapped
  later without touching batching/merging/validation logic.
- Every LLM response is validated against the expected schema before being
  trusted. LLMs routinely wrap JSON in markdown fences or add stray text —
  this is handled explicitly, not assumed away.
- Confidence scores are never invented past what evidence supports: merging
  multiple batches' opinions on the same trait uses an evidence-weighted
  average, so a trait seen once stays low-confidence even if the LLM itself
  reported a high number for that one batch.

Usage:
    python profile_builder.py path/to/user_messages.json

Requires:
    pip install google-generativeai
    Set the GEMINI_API_KEY environment variable before running.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1000        # messages per LLM call. Gemini Flash supports up to a
                         # ~1M token context window, so large batches are safe and
                         # give the model far more context to spot real patterns
                         # (vs. isolated snapshots from tiny batches). Lower this
                         # only if you see truncation or context-limit errors.
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5
MAX_BATCH_CHARS = 400_000    # ~100k tokens. Batches are capped by size, not just message
                             # count, because a few long messages can push a 2000-message
                             # batch past the free tier's 250k tokens/minute limit.
CHECKPOINT_DIR = Path("batch_checkpoints")
BATCH_COOLDOWN_SECONDS = 65  # free tier caps input at ~250k tokens/minute; pausing
                             # between large batches keeps each one in its own window


class ProfileBuilderError(Exception):
    """Base error for this module."""


class LLMResponseError(ProfileBuilderError):
    """Raised when an LLM response cannot be parsed into valid, schema-matching JSON."""


# ---------------------------------------------------------------------- #
# LLM provider — isolated so it can be swapped without touching the rest
# ---------------------------------------------------------------------- #

class LLMProvider:
    """Thin wrapper around the Gemini REST API (no SDK dependency — avoids grpc,
    which requires a native DLL that some locked-down Windows environments block
    via Application Control policy)."""

    API_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ProfileBuilderError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment variable."
            )
        self.model_name = model_name
        try:
            import requests  # noqa: F401 — just confirming it's available
        except ImportError as e:
            raise ProfileBuilderError(
                "The 'requests' library is not installed. Run: pip install requests"
            ) from e

    def generate(self, prompt: str) -> str:
        """Calls the LLM once via REST and returns the raw text response. Raises on total failure."""
        import requests

        url = self.API_URL_TEMPLATE.format(model=self.model_name)
        # Key goes in a header, NOT the URL: requests prints the full URL in
        # connection-error messages, which would leak the key into logs.
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=300
                )
                if response.status_code != 200:
                    raise LLMResponseError(
                        f"Gemini API returned status {response.status_code}: {response.text[:500]}"
                    )
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMResponseError(f"No candidates in Gemini response: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    raise LLMResponseError("Gemini returned an empty response.")
                return text
            except (requests.RequestException, LLMResponseError) as e:
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise LLMResponseError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------- #
# Response parsing / validation
# ---------------------------------------------------------------------- #

def extract_json_from_llm_text(raw_text: str) -> dict[str, Any]:
    """
    LLMs frequently wrap JSON in ```json ... ``` fences, or add a sentence
    before/after the JSON block. This strips that noise and parses what's left.
    Raises LLMResponseError with the raw text included if parsing still fails,
    so failures are debuggable rather than silent.
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # No fence — fall back to grabbing the first {...} block found anywhere in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(
            f"Could not parse LLM response as JSON: {e}\n--- Raw response ---\n{raw_text[:500]}"
        ) from e


REQUIRED_TOP_LEVEL_KEYS = {
    "communication_preferences",
    "explanation_preferences",
    "learning_evidence",
}


def validate_profile_schema(data: dict[str, Any]) -> None:
    """
    Validates that a parsed batch response has the expected shape.
    Raises LLMResponseError with a specific reason if it doesn't — callers
    should treat this the same as a parse failure (i.e. retry or skip the batch).
    """
    if not isinstance(data, dict):
        raise LLMResponseError(f"Expected a JSON object, got {type(data).__name__}.")

    missing = REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise LLMResponseError(f"Response is missing required key(s): {missing}")

    if not isinstance(data["learning_evidence"], list):
        raise LLMResponseError(
            f"'learning_evidence' must be a list, got {type(data['learning_evidence']).__name__}."
        )

    for section_name in ("communication_preferences", "explanation_preferences"):
        section = data[section_name]
        if not isinstance(section, dict):
            raise LLMResponseError(f"'{section_name}' must be an object.")
        for trait_name, trait_value in section.items():
            if not isinstance(trait_value, dict) or "confidence" not in trait_value:
                raise LLMResponseError(
                    f"Trait '{trait_name}' in '{section_name}' is missing a 'confidence' field."
                )


# ---------------------------------------------------------------------- #
# Batching
# ---------------------------------------------------------------------- #

def load_user_messages(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ProfileBuilderError(f"Input file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ProfileBuilderError(f"'{path}' is not valid JSON: {e}") from e

    if not isinstance(data, list) or not data:
        raise ProfileBuilderError(f"'{path}' does not contain a non-empty list of messages.")
    return data


def make_batches(
    messages: list[dict[str, Any]],
    batch_size: int,
    max_chars: int = MAX_BATCH_CHARS,
) -> list[list[dict[str, Any]]]:
    """Groups messages into batches capped by BOTH message count and total characters,
    whichever limit is hit first. Order is preserved."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for msg in messages:
        msg_chars = len(msg.get("content", ""))
        if current and (len(current) >= batch_size or current_chars + msg_chars > max_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(msg)
        current_chars += msg_chars
    if current:
        batches.append(current)
    return batches


def batch_checkpoint_path(batch: list[dict[str, Any]]) -> Path:
    """Stable filename derived from the batch's contents, so a rerun recognises
    batches that are already done even if batch numbering changes."""
    fingerprint = hashlib.sha1(
        "|".join(str(m.get("message_id", "")) for m in batch).encode("utf-8")
    ).hexdigest()[:12]
    return CHECKPOINT_DIR / f"batch_{fingerprint}.json"


def build_batch_prompt(batch: list[dict[str, Any]]) -> str:
    message_texts = "\n---\n".join(m.get("content", "") for m in batch)
    return f"""You are analyzing a student's messages to an AI assistant to infer how they learn.

Below are {len(batch)} messages the student wrote, separated by "---".

Based ONLY on these messages, infer:
1. communication_preferences: verbosity (low/moderate/high), directness (low/moderate/high)
2. explanation_preferences: step_by_step (true/false), concrete_examples (true/false)
3. learning_evidence: a list of concepts the student engaged with, each with a "state"
   (e.g. "frequently practiced", "asked about once", "showed confusion about"),
   a confidence score (0.0-1.0), and evidence_count (how many messages in THIS batch
   support it).

Every trait must include a "confidence" field (0.0 to 1.0) reflecting how much evidence
in THIS batch supports it. Do not report a trait you have no evidence for.

Respond with ONLY a single JSON object, no explanation text, in exactly this shape:
{{
  "communication_preferences": {{
    "verbosity": {{"value": "...", "confidence": 0.0}},
    "directness": {{"value": "...", "confidence": 0.0}}
  }},
  "explanation_preferences": {{
    "step_by_step": {{"value": true, "confidence": 0.0}},
    "concrete_examples": {{"value": true, "confidence": 0.0}}
  }},
  "learning_evidence": [
    {{"concept": "...", "state": "...", "confidence": 0.0, "evidence_count": 0}}
  ]
}}

Messages:
{message_texts}
"""


# ---------------------------------------------------------------------- #
# Merging partial profiles
# ---------------------------------------------------------------------- #

def _weighted_merge_trait(existing: Optional[dict], new: dict, new_weight: int) -> dict:
    """Merges one trait's value across batches using an evidence-weighted average
    for confidence. Newer batches don't overwrite older ones blindly — they're blended."""
    if existing is None:
        return {**new, "_weight": new_weight}

    old_weight = existing.get("_weight", 1)
    total_weight = old_weight + new_weight
    merged_confidence = (
        existing["confidence"] * old_weight + new["confidence"] * new_weight
    ) / total_weight

    # Keep whichever value came from the higher-confidence batch
    value = new["value"] if new["confidence"] >= existing["confidence"] else existing["value"]

    return {"value": value, "confidence": round(merged_confidence, 3), "_weight": total_weight}


def merge_batch_results(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged_comm: dict[str, dict] = {}
    merged_expl: dict[str, dict] = {}
    evidence_by_concept: dict[str, dict[str, Any]] = {}

    for result in batch_results:
        for trait_name, trait_value in result.get("communication_preferences", {}).items():
            weight = trait_value.get("confidence", 0) > 0 and 1 or 0
            merged_comm[trait_name] = _weighted_merge_trait(
                merged_comm.get(trait_name), trait_value, new_weight=max(weight, 1)
            )

        for trait_name, trait_value in result.get("explanation_preferences", {}).items():
            merged_expl[trait_name] = _weighted_merge_trait(
                merged_expl.get(trait_name), trait_value, new_weight=1
            )

        for evidence in result.get("learning_evidence", []):
            concept = evidence.get("concept", "unknown")
            count = evidence.get("evidence_count", 1)
            if concept not in evidence_by_concept:
                evidence_by_concept[concept] = {
                    "concept": concept,
                    "state": evidence.get("state", ""),
                    "confidence": evidence.get("confidence", 0.0),
                    "evidence_count": count,
                }
            else:
                existing = evidence_by_concept[concept]
                total = existing["evidence_count"] + count
                existing["confidence"] = round(
                    (existing["confidence"] * existing["evidence_count"]
                     + evidence.get("confidence", 0.0) * count) / total,
                    3,
                )
                existing["evidence_count"] = total
                # more evidence generally means more certainty about the state, not less
                if count > existing["evidence_count"] - count:
                    existing["state"] = evidence.get("state", existing["state"])

    def strip_weight(d: dict) -> dict:
        return {k: {"value": v["value"], "confidence": v["confidence"]} for k, v in d.items()}

    return {
        "communication_preferences": strip_weight(merged_comm),
        "explanation_preferences": strip_weight(merged_expl),
        "learning_evidence": sorted(
            evidence_by_concept.values(), key=lambda e: e["evidence_count"], reverse=True
        ),
    }


# ---------------------------------------------------------------------- #
# Orchestration
# ---------------------------------------------------------------------- #

def build_profile(messages_path: Path, llm: LLMProvider) -> dict[str, Any]:
    messages = load_user_messages(messages_path)
    batches = make_batches(messages, BATCH_SIZE)
    logger.info(
        "Processing %d messages in %d batches (max %d messages / %d chars each).",
        len(messages), len(batches), BATCH_SIZE, MAX_BATCH_CHARS,
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    batch_results: list[dict[str, Any]] = []
    failed_batches = 0
    made_api_call = False

    for i, batch in enumerate(batches, start=1):
        checkpoint = batch_checkpoint_path(batch)

        # Resume support: a batch that already succeeded on an earlier run is
        # loaded from disk instead of being sent to the LLM again.
        if checkpoint.exists():
            try:
                with open(checkpoint, "r", encoding="utf-8") as f:
                    batch_results.append(json.load(f))
                logger.info("Batch %d/%d loaded from checkpoint (skipping API call).", i, len(batches))
                continue
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Checkpoint %s unreadable (%s); redoing batch.", checkpoint.name, e)

        if made_api_call:
            logger.info("Cooling down %ds to stay under the token-per-minute limit...", BATCH_COOLDOWN_SECONDS)
            time.sleep(BATCH_COOLDOWN_SECONDS)

        logger.info("Processing batch %d/%d (%d messages)...", i, len(batches), len(batch))
        prompt = build_batch_prompt(batch)
        made_api_call = True
        try:
            raw_response = llm.generate(prompt)
            parsed = extract_json_from_llm_text(raw_response)
            validate_profile_schema(parsed)
        except LLMResponseError as e:
            failed_batches += 1
            logger.error("Batch %d failed and will be skipped this run: %s", i, str(e)[:300])
            continue

        batch_results.append(parsed)
        with open(checkpoint, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False)  # saved immediately: a crash or Ctrl+C can't lose it

    if not batch_results:
        raise ProfileBuilderError(
            f"All {len(batches)} batches failed. No profile could be built. Check the logs above."
        )

    if failed_batches:
        logger.warning(
            "%d of %d batches failed. The profile below is PARTIAL. "
            "Re-run the same command to retry only the failed batches "
            "(finished ones are reloaded from '%s').",
            failed_batches, len(batches), CHECKPOINT_DIR,
        )

    logger.info(
        "Merging %d successful batch(es) (%d failed) into final profile.",
        len(batch_results), failed_batches,
    )
    return merge_batch_results(batch_results)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python profile_builder.py <path_to_user_messages.json>")
        sys.exit(1)

    messages_path = Path(sys.argv[1])

    try:
        llm = LLMProvider()
        profile = build_profile(messages_path, llm)
    except ProfileBuilderError as e:
        logger.error(str(e))
        sys.exit(1)

    out_path = Path("learner_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"\nProfile saved to {out_path.resolve()}")
    print(json.dumps(profile, indent=2)[:1000])


if __name__ == "__main__":
    main()
