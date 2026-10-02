import pytest

from app import (
    ProfileUploadError,
    generate_pdf_from_bytes,
    generate_profile_from_conversation_upload,
)


VALID_PROFILE = {
    "dataset_summary": {"selected_conversations": 1},
    "topic_activity": [
        {"topic": "Python", "conversation_count": 1, "evidence_ids": ["E001"]}
    ],
    "observed_patterns": [
        {
            "pattern": "Asks about Python errors",
            "observation": "A message asks for help understanding an import error.",
            "evidence_ids": ["E001"],
        }
    ],
    "explicit_preferences": [],
    "not_inferable_from_chat_history": ["independent mastery"],
    "evidence": [
        {
            "evidence_id": "E001",
            "text": "My Python code gives an import error. What does this mean?",
            "conversation_title": "Python debugging",
            "source_note": "short_user_turn",
            "domains": ["Python"],
        }
    ],
}


def test_uploaded_profile_renders_a_pdf():
    import json

    profile, pdf_bytes = generate_pdf_from_bytes(json.dumps(VALID_PROFILE).encode("utf-8"))

    assert profile["topic_activity"][0]["topic"] == "Python"
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"/Type /Pages" in pdf_bytes


def test_pdf_topic_table_shows_counts_without_literal_html_tags():
    from reportlab.platypus import Table

    from generate_profile_pdf import build_topics, make_styles

    story = build_topics(VALID_PROFILE, make_styles())
    table = next(item for item in story if isinstance(item, Table))
    topic_cell = table._cellvalues[1][0]
    conversation_cell = table._cellvalues[1][1]
    evidence_cell = table._cellvalues[1][2]

    assert topic_cell.text == "Python"
    assert conversation_cell.text == "1"
    assert evidence_cell.text == "1"


def test_pdf_evidence_record_uses_each_message_and_its_source_metadata():
    from reportlab.platypus import LongTable

    from generate_profile_pdf import build_evidence_appendix, make_styles

    story = build_evidence_appendix(VALID_PROFILE, make_styles())
    table = next(item for item in story if isinstance(item, LongTable))
    rows = table._cellvalues[1:]

    assert len(rows) == len(VALID_PROFILE["evidence"])
    assert "E001" in rows[0][0].text
    assert "Python debugging" in rows[0][0].text
    assert rows[0][1].text == VALID_PROFILE["evidence"][0]["text"]


def test_uploaded_profile_rejects_missing_evidence_reference():
    import json

    invalid_profile = {
        **VALID_PROFILE,
        "observed_patterns": [
            {"pattern": "Uncited claim", "observation": "No source", "evidence_ids": ["E999"]}
        ],
    }

    with pytest.raises(ProfileUploadError, match="missing evidence"):
        generate_pdf_from_bytes(json.dumps(invalid_profile).encode("utf-8"))


def test_uploaded_profile_rejects_invalid_json():
    with pytest.raises(ProfileUploadError, match="not valid UTF-8 JSON"):
        generate_pdf_from_bytes(b"not-json")


def test_staged_upload_is_used_once_for_profile_generation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module

    staged_dir = tmp_path / "staged-uploads"
    monkeypatch.setattr(app_module, "STAGED_UPLOAD_DIR", staged_dir)
    generated = {}

    def fake_generate_profile(*, uploaded_bytes, filename, api_key=None):
        generated["bytes"] = uploaded_bytes
        generated["filename"] = filename
        return {}, b"%PDF-test", {}

    monkeypatch.setattr(
        app_module,
        "generate_profile_from_conversation_upload",
        fake_generate_profile,
    )

    with TestClient(app_module.app) as client:
        upload_response = client.post(
            "/upload",
            files={"file": ("chat-export.zip", b"zip contents", "application/zip")},
        )
        assert upload_response.status_code == 200
        upload_id = upload_response.json()["upload_id"]
        assert list(staged_dir.glob("*.zip"))

        pdf_response = client.post("/analyze", data={"upload_id": upload_id})
        assert pdf_response.status_code == 200
        assert pdf_response.content.startswith(b"%PDF-")

        expired_response = client.post("/analyze", data={"upload_id": upload_id})
        assert expired_response.status_code == 410

    assert generated == {"bytes": b"zip contents", "filename": "chat-export.zip"}
    assert not list(staged_dir.glob("*.zip"))


class FakeProfileLLM:
    def generate(self, prompt):
        import json

        return json.dumps({
            "observed_patterns": [
                {
                    "pattern": "Returns with a follow-up debugging question",
                    "observation": "Two messages describe a Python import error and a follow-up attempt.",
                    "evidence_ids": ["E001", "E002"],
                }
            ],
            "explicit_preferences": [],
        })


def _sample_messages():
    return [
        {
            "conversation_id": "conversation-1",
            "conversation_title": "Python import debugging",
            "message_id": "message-1",
            "content": "My Python code gives an import error. How do I fix this?",
            "create_time": 1.0,
        },
        {
            "conversation_id": "conversation-1",
            "conversation_title": "Python import debugging",
            "message_id": "message-2",
            "content": "I tried changing the Python import, but it still gives an error. Why?",
            "create_time": 2.0,
        },
    ]


def test_full_pipeline_accepts_extracted_user_messages_json():
    import json

    profile, pdf_bytes, stats = generate_profile_from_conversation_upload(
        json.dumps(_sample_messages()).encode("utf-8"),
        "user_messages.json",
        llm=FakeProfileLLM(),
    )

    assert stats["messages"] == 2
    assert stats["selected_conversations"] == 1
    assert profile["topic_activity"][0]["topic"] == "Python"
    assert profile["observed_patterns"][0]["evidence_ids"] == ["E001", "E002"]
    assert pdf_bytes.startswith(b"%PDF-")


def test_full_pipeline_extracts_chatgpt_export_zip():
    import io
    import json
    import zipfile

    user_texts = [
        "My Python code gives an import error. How do I fix this?",
        "I tried changing the Python import, but it still gives an error. Why?",
    ]
    mapping = {}
    parent = None
    for index, text in enumerate(user_texts, start=1):
        node_id = f"node-{index}"
        mapping[node_id] = {
            "parent": parent,
            "message": {
                "author": {"role": "user"},
                "content": {"parts": [text]},
                "create_time": float(index),
            },
        }
        parent = node_id
    conversation = {
        "id": "conversation-1",
        "title": "Python import debugging",
        "mapping": mapping,
        "current_node": "node-2",
    }

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("conversations-000.json", json.dumps([conversation]))

    profile, pdf_bytes, stats = generate_profile_from_conversation_upload(
        archive.getvalue(),
        "chatgpt-export.zip",
        llm=FakeProfileLLM(),
    )

    assert stats["messages"] == 2
    assert profile["topic_activity"][0]["topic"] == "Python"
    assert pdf_bytes.startswith(b"%PDF-")
