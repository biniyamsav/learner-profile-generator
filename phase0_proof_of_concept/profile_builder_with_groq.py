import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5
BATCH_COOLDOWN_SECONDS = 3  # Cooldown between batches
DEFAULT_GROQ_MODELS = (
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
)

# Schema required by profile validator
EXPECTED_KEYS = {
    "technical_skills",
    "soft_skills",
    "learning_goals",
    "frequently_asked_topics",
    "communication_style_preference",
    "difficulty_level",
    "background_context",
    "sentiment_summary",
}


class ProfileBuilderError(Exception):
    """Base exception for profile builder errors."""
    pass


class LLMResponseError(ProfileBuilderError):
    """Raised when the LLM returns invalid or empty response."""
    pass


class LLMProvider:
    """Thin wrapper around the Groq REST API (OpenAI compatible format)."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ProfileBuilderError(
                "No Groq API key found. Set the GROQ_API_KEY environment variable."
            )

        preferred_model = model_name or os.environ.get("GROQ_MODEL")
        fallback_models = [
            preferred_model,
            *[m for m in DEFAULT_GROQ_MODELS if m != preferred_model],
        ]
        self.model_candidates = [m for m in fallback_models if m]
        self.model_name = self.model_candidates[0] if self.model_candidates else DEFAULT_GROQ_MODELS[0]

    def generate(self, prompt: str) -> str:
        """Calls Groq via REST API and returns raw text response."""
        try:
            import requests
        except ImportError as e:
            raise ProfileBuilderError(
                "The 'requests' library is not installed. Run: pip install requests"
            ) from e

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error: Optional[Exception] = None

        for model_name in self.model_candidates:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = requests.post(
                        self.API_URL, headers=headers, json=payload, timeout=90
                    )

                    if response.status_code == 429:
                        wait_time = 30 * attempt
                        logger.warning(
                            "Groq rate limit hit (429). Waiting %d seconds (attempt %d/%d)...",
                            wait_time, attempt, MAX_RETRIES
                        )
                        time.sleep(wait_time)
                        continue

                    if response.status_code == 404 or response.status_code == 400:
                        body = response.text[:500]
                        if "model" in body.lower() or "decommissioned" in body.lower():
                            logger.warning(
                                "Model %s is unavailable on this Groq account (%s). Trying next candidate.",
                                model_name,
                                body,
                            )
                            break

                    if response.status_code != 200:
                        raise LLMResponseError(
                            f"Groq API returned status {response.status_code}: {response.text[:500]}"
                        )

                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMResponseError(f"No choices returned in Groq response: {data}")

                    text = choices[0].get("message", {}).get("content", "").strip()
                    if not text:
                        raise LLMResponseError("Groq returned an empty response.")

                    return text

                except (requests.RequestException, LLMResponseError) as e:
                    last_error = e
                    logger.warning(
                        "LLM call failed for model %s (attempt %d/%d): %s",
                        model_name, attempt, MAX_RETRIES, e
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

            if model_name != self.model_candidates[-1]:
                continue

        raise LLMResponseError(f"LLM call failed after {MAX_RETRIES} attempts for all Groq models: {last_error}")


def load_user_messages(filepath: Path) -> List[Dict[str, Any]]:
    """Loads cleaned user messages from file."""
    if not filepath.exists():
        raise ProfileBuilderError(f"Messages file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("messages", [])


def make_batches(messages: List[Dict[str, Any]], batch_size: int = 1000) -> List[List[Dict[str, Any]]]:
    """Splits input messages into lists of specified batch size."""
    return [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]


def build_batch_prompt(batch: List[Dict[str, Any]]) -> str:
    """Builds the extraction prompt for a single batch of messages."""
    messages_text = "\n".join(
        f"- [{msg.get('timestamp', 'N/A')}] {msg.get('text', '')}" for msg in batch
    )
    return f"""Analyze the following user chat history batch and extract insights into a structured JSON profile.

Return ONLY a valid JSON object matching this schema:
{{
  "technical_skills": ["list of strings"],
  "soft_skills": ["list of strings"],
  "learning_goals": ["list of strings"],
  "frequently_asked_topics": ["list of strings"],
  "communication_style_preference": "string",
  "difficulty_level": "string",
  "background_context": "string",
  "sentiment_summary": "string"
}}

User Messages Batch:
{messages_text}
"""


def extract_json(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON from raw LLM output text."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def validate_profile_schema(data: Dict[str, Any]) -> bool:
    """Validates that extracted profile JSON matches the expected schema keys."""
    if not isinstance(data, dict):
        return False
    return EXPECTED_KEYS.issubset(data.keys())


def merge_batch_profiles(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merges checkpoint profiles from multiple batches into a consolidated profile."""
    merged = {
        "technical_skills": set(),
        "soft_skills": set(),
        "learning_goals": set(),
        "frequently_asked_topics": set(),
        "communication_style_preference": [],
        "difficulty_level": [],
        "background_context": [],
        "sentiment_summary": [],
    }

    for p in profiles:
        for key in ["technical_skills", "soft_skills", "learning_goals", "frequently_asked_topics"]:
            merged[key].update(p.get(key, []))
        for key in ["communication_style_preference", "difficulty_level", "background_context", "sentiment_summary"]:
            val = p.get(key)
            if val:
                merged[key].append(val)

    return {
        "technical_skills": sorted(list(merged["technical_skills"])),
        "soft_skills": sorted(list(merged["soft_skills"])),
        "learning_goals": sorted(list(merged["learning_goals"])),
        "frequently_asked_topics": sorted(list(merged["frequently_asked_topics"])),
        "communication_style_preference": " | ".join(set(merged["communication_style_preference"])),
        "difficulty_level": " | ".join(set(merged["difficulty_level"])),
        "background_context": " | ".join(set(merged["background_context"])),
        "sentiment_summary": " | ".join(set(merged["sentiment_summary"])),
    }


def main():
    base_dir = Path(__file__).parent
    messages_file = base_dir / "user_messages.json"
    checkpoints_dir = base_dir / "batch_checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    output_profile_file = base_dir / "learner_profile.json"

    logger.info("Loading user messages...")
    messages = load_user_messages(messages_file)
    logger.info("Loaded %d total messages.", len(messages))

    batches = make_batches(messages, batch_size=1000)
    logger.info("Created %d batches.", len(batches))

    provider = LLMProvider(model_name=os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b")
    parsed_profiles = []

    for idx, batch in enumerate(batches, start=1):
        checkpoint_path = checkpoints_dir / f"batch_{idx:02d}.json"

        if checkpoint_path.exists():
            logger.info("Batch %d/%d checkpoint found. Loading...", idx, len(batches))
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                parsed_profiles.append(json.load(f))
            continue

        logger.info("Processing Batch %d/%d (%d messages)...", idx, len(batches), len(batch))
        prompt = build_batch_prompt(batch)

        try:
            raw_response = provider.generate(prompt)
            parsed = extract_json(raw_response)

            if not validate_profile_schema(parsed):
                raise ProfileBuilderError("Profile schema validation failed.")

            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)

            parsed_profiles.append(parsed)
            logger.info("Batch %d/%d completed and saved.", idx, len(batches))

        except Exception as e:
            logger.error("Failed processing batch %d: %s", idx, e)
            raise

        time.sleep(BATCH_COOLDOWN_SECONDS)

    logger.info("Consolidating batch profile results...")
    final_profile = merge_batch_profiles(parsed_profiles)

    with open(output_profile_file, "w", encoding="utf-8") as f:
        json.dump(final_profile, f, ensure_ascii=False, indent=2)

    logger.info("Successfully saved consolidated profile to %s", output_profile_file)


if __name__ == "__main__":
    main()