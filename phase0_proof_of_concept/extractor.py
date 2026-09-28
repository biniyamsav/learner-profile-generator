"""
Extracts only user-authored messages from an OpenAI ChatGPT history export.

Accepts three input shapes:
  - a .zip file containing conversations*.json chunk(s)
  - a single conversations*.json file
  - a folder containing loose conversations-*.json chunk files
    (e.g. conversations-000.json ... conversations-009.json)

Usage:
    python chatgpt_user_message_extractor.py path/to/conversations.zip
    python chatgpt_user_message_extractor.py path/to/conversations.json
    python chatgpt_user_message_extractor.py path/to/chatgpt_history/
"""

from __future__ import annotations

import json
import logging
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when the export cannot be parsed into usable conversation data."""


@dataclass(frozen=True)
class UserMessage:
    conversation_id: str
    conversation_title: str
    message_id: str
    content: str
    create_time: Optional[float]


class ChatGPTUserMessageExtractor:
    """
    Parses a ChatGPT export and extracts only messages authored by the user.

    Design notes:
    - Every malformed record is skipped and logged, never allowed to crash
      the whole run — a single corrupted conversation shouldn't lose the
      other 999.
    - Extraction logic (parts -> text) is isolated so it can be unit tested
      independently of file I/O.
    - Returns dataclass instances, not raw dicts, so downstream code gets
      type safety instead of guessing at key names.
    """

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source path not found at: {self.file_path}")
        if self.file_path.is_file() and self.file_path.suffix not in (".zip", ".json"):
            raise ValueError(
                f"Unsupported file format '{self.file_path.suffix}'. "
                "Expected a .zip, .json file, or a folder of conversations-*.json chunks."
            )

    # ------------------------------------------------------------------ #
    # Loading raw conversation records
    # ------------------------------------------------------------------ #

    def _load_raw_conversations(self) -> list[dict[str, Any]]:
        if self.file_path.is_dir():
            return self._load_from_folder()
        if self.file_path.suffix == ".zip":
            return self._load_from_zip()
        return self._load_from_json_file(self.file_path)

    def _load_from_folder(self) -> list[dict[str, Any]]:
        """Loads and merges every conversations*.json chunk in a folder, sorted by filename
        so conversations-000.json is processed before conversations-001.json, etc."""
        chunk_files = sorted(
            f for f in self.file_path.glob("*.json")
            if "conversations" in f.stem.lower()
        )
        if not chunk_files:
            raise ParseError(
                f"No conversations*.json files found in folder: {self.file_path}. "
                f"Files present: {[f.name for f in self.file_path.iterdir() if f.is_file()]}"
            )

        all_conversations: list[dict[str, Any]] = []
        for chunk_path in chunk_files:
            logger.info("Loading chunk: %s", chunk_path.name)
            try:
                chunk_data = self._load_from_json_file(chunk_path)
                all_conversations.extend(chunk_data)
            except ParseError as e:
                logger.error("Skipping unreadable chunk %s: %s", chunk_path.name, e)
                continue

        logger.info(
            "Merged %d total conversation records from %d chunk file(s).",
            len(all_conversations),
            len(chunk_files),
        )
        return all_conversations

    def _load_from_zip(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        try:
            with zipfile.ZipFile(self.file_path, "r") as archive:
                target_files = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".json") and "conversations" in Path(name).stem.lower()
                ]
                if not target_files:
                    raise ParseError(
                        f"No conversations*.json file found inside {self.file_path.name}. "
                        f"Archive contains: {archive.namelist()[:10]}"
                    )
                for filename in target_files:
                    logger.info("Loading chunk: %s", filename)
                    try:
                        with archive.open(filename) as f:
                            data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error("Skipping unreadable chunk %s: %s", filename, e)
                        continue
                    if isinstance(data, list):
                        conversations.extend(data)
                    else:
                        logger.warning(
                            "Chunk %s did not contain a list at the top level; skipped.",
                            filename,
                        )
        except zipfile.BadZipFile as e:
            raise ParseError(f"'{self.file_path}' is not a valid zip archive: {e}") from e

        logger.info("Loaded %d raw conversation records from zip.", len(conversations))
        return conversations

    def _load_from_json_file(self, path: Path) -> list[dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ParseError(f"'{path}' is not valid JSON: {e}") from e

        if not isinstance(data, list):
            raise ParseError(
                f"Expected a top-level list of conversations in '{path}', "
                f"got {type(data).__name__}."
            )
        logger.info("Loaded %d raw conversation records from JSON.", len(data))
        return data

    # ------------------------------------------------------------------ #
    # Text extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text_from_parts(parts: Any) -> str:
        """Safely extract text from a message's 'parts' field, whatever shape it's in."""
        if not isinstance(parts, list):
            return ""
        fragments = []
        for part in parts:
            if isinstance(part, str):
                fragments.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                fragments.append(part["text"])
            # Silently skip other part types (images, tool calls, etc.) —
            # this extractor is text-only by design.
        return "\n".join(fragments).strip()

    # ------------------------------------------------------------------ #
    # Per-conversation parsing
    # ------------------------------------------------------------------ #

    def _extract_user_messages_from_conversation(
        self, raw_conv: dict[str, Any]
    ) -> list[UserMessage]:
        conv_id = str(raw_conv.get("conversation_id") or raw_conv.get("id") or "unknown_id")
        title = raw_conv.get("title") or "Untitled Conversation"
        mapping = raw_conv.get("mapping")

        if not isinstance(mapping, dict):
            logger.warning("Conversation '%s' has no valid mapping; skipped.", conv_id)
            return []

        current_node_id = raw_conv.get("current_node")
        visited: set[str] = set()
        user_messages: list[UserMessage] = []

        while current_node_id and current_node_id in mapping and current_node_id not in visited:
            visited.add(current_node_id)
            node = mapping.get(current_node_id) or {}
            msg_data = node.get("message")

            if isinstance(msg_data, dict):
                role = (msg_data.get("author") or {}).get("role")
                content = msg_data.get("content") or {}
                parts = content.get("parts")
                text = self._extract_text_from_parts(parts)

                if role == "user" and text:
                    user_messages.append(
                        UserMessage(
                            conversation_id=conv_id,
                            conversation_title=title,
                            message_id=str(current_node_id),
                            content=text,
                            create_time=msg_data.get("create_time"),
                        )
                    )

            current_node_id = node.get("parent")

        user_messages.reverse()  # chronological order
        return user_messages

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_all_user_messages(self) -> list[UserMessage]:
        """Returns every user-authored message across the whole export, chronological per conversation."""
        raw_conversations = self._load_raw_conversations()
        all_messages: list[UserMessage] = []
        skipped = 0

        for raw_conv in raw_conversations:
            try:
                messages = self._extract_user_messages_from_conversation(raw_conv)
                all_messages.extend(messages)
            except Exception as e:  # noqa: BLE001 — deliberately broad: one bad record must not kill the run
                skipped += 1
                logger.error("Failed to parse a conversation, skipping it: %s", e)

        logger.info(
            "Extracted %d user messages from %d conversations (%d conversations failed to parse).",
            len(all_messages),
            len(raw_conversations),
            skipped,
        )
        return all_messages


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python chatgpt_user_message_extractor.py <path_to_export.zip|.json>")
        sys.exit(1)

    extractor = ChatGPTUserMessageExtractor(sys.argv[1])
    messages = extractor.extract_all_user_messages()

    if not messages:
        print("No user messages were extracted. Check the export file and logs above.")
        return

    print(f"\n--- SAMPLE ({min(3, len(messages))} of {len(messages)} messages) ---")
    for msg in messages[:3]:
        preview = msg.content[:150].replace("\n", " ")
        print(f"[{msg.conversation_title}] {preview}...")

    out_path = Path("user_messages.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in messages], f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(messages)} messages to {out_path.resolve()}")


if __name__ == "__main__":
    main()