import pytest
from profile_builder import (
    extract_json_from_llm_text,
    validate_profile_schema,
    LLMResponseError,
)

VALID_PROFILE = {
    "communication_preferences": {
        "verbosity": {"value": "moderate", "confidence": 0.7}
    },
    "explanation_preferences": {
        "step_by_step": {"value": True, "confidence": 0.8}
    },
    "learning_evidence": [
        {"concept": "SQL joins", "state": "practiced", "confidence": 0.6, "evidence_count": 3}
    ],
}


# ------------------------------------------------------------------ #
# extract_json_from_llm_text
# ------------------------------------------------------------------ #

def test_extract_json_handles_clean_json():
    text = '{"a": 1}'
    assert extract_json_from_llm_text(text) == {"a": 1}


def test_extract_json_handles_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert extract_json_from_llm_text(text) == {"a": 1}


def test_extract_json_handles_fence_without_json_label():
    text = '```\n{"a": 1}\n```'
    assert extract_json_from_llm_text(text) == {"a": 1}


def test_extract_json_handles_extra_text_around_json():
    text = 'Sure, here is the result:\n{"a": 1}\nLet me know if you need more.'
    assert extract_json_from_llm_text(text) == {"a": 1}


def test_extract_json_raises_on_garbage_input():
    with pytest.raises(LLMResponseError):
        extract_json_from_llm_text("this is not json at all")


def test_extract_json_raises_on_empty_string():
    with pytest.raises(LLMResponseError):
        extract_json_from_llm_text("")


# ------------------------------------------------------------------ #
# validate_profile_schema
# ------------------------------------------------------------------ #

def test_validate_schema_accepts_valid_profile():
    validate_profile_schema(VALID_PROFILE)  # should not raise


def test_validate_schema_rejects_non_dict():
    with pytest.raises(LLMResponseError):
        validate_profile_schema(["not", "a", "dict"])


def test_validate_schema_rejects_missing_top_level_key():
    broken = {k: v for k, v in VALID_PROFILE.items() if k != "learning_evidence"}
    with pytest.raises(LLMResponseError):
        validate_profile_schema(broken)


def test_validate_schema_rejects_learning_evidence_not_a_list():
    broken = {**VALID_PROFILE, "learning_evidence": {"not": "a list"}}
    with pytest.raises(LLMResponseError):
        validate_profile_schema(broken)


def test_validate_schema_rejects_trait_missing_confidence():
    broken = {
        **VALID_PROFILE,
        "communication_preferences": {"verbosity": {"value": "moderate"}},  # no confidence
    }
    with pytest.raises(LLMResponseError):
        validate_profile_schema(broken)


# ------------------------------------------------------------------ #
# make_batches / checkpointing / resume
# ------------------------------------------------------------------ #

import json
import profile_builder
from profile_builder import make_batches, build_profile, LLMResponseError


def _msgs(n, chars=10):
    return [{"message_id": f"m{i}", "content": "x" * chars} for i in range(n)]


def test_make_batches_respects_message_count_cap():
    batches = make_batches(_msgs(10), batch_size=4, max_chars=10_000)
    assert [len(b) for b in batches] == [4, 4, 2]


def test_make_batches_respects_character_cap():
    # 10 messages x 100 chars, cap of 250 chars -> 2 messages per batch
    batches = make_batches(_msgs(10, chars=100), batch_size=1000, max_chars=250)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_make_batches_keeps_oversized_single_message():
    batches = make_batches(_msgs(1, chars=5000), batch_size=10, max_chars=100)
    assert len(batches) == 1 and len(batches[0]) == 1


class FakeLLM:
    def __init__(self, fail_first_n=0):
        self.calls = 0
        self.fail_first_n = fail_first_n

    def generate(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise LLMResponseError("simulated failure")
        return json.dumps(VALID_PROFILE)


def _setup(tmp_path, monkeypatch, n_messages=6):
    monkeypatch.setattr(profile_builder, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(profile_builder, "BATCH_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(profile_builder, "BATCH_SIZE", 2)  # 6 messages -> 3 batches
    path = tmp_path / "msgs.json"
    path.write_text(json.dumps(_msgs(n_messages)), encoding="utf-8")
    return path


def test_resume_skips_batches_that_already_succeeded(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch)

    first = FakeLLM()
    build_profile(path, first)
    assert first.calls == 3

    second = FakeLLM()
    build_profile(path, second)
    assert second.calls == 0  # everything came from checkpoints


def test_failed_batch_is_retried_on_rerun_without_redoing_others(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch)

    flaky = FakeLLM(fail_first_n=1)  # batch 1 fails, batches 2-3 succeed
    build_profile(path, flaky)
    assert flaky.calls == 3

    rerun = FakeLLM()
    build_profile(path, rerun)
    assert rerun.calls == 1  # only the one failed batch is sent again


def test_all_batches_failing_raises(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch)
    with pytest.raises(profile_builder.ProfileBuilderError):
        build_profile(path, FakeLLM(fail_first_n=99))


def test_build_profile_accepts_compact_context(tmp_path):
    compact = {
        "dataset_summary": {"total_messages": 10, "selected_conversations": 2},
        "learner_profile_context": {
            "main_domains": ["Python", "SQL"],
            "learning_behaviors": ["debugging", "project_building"],
            "domain_conversation_counts": {"Python": 1, "SQL": 1},
            "conversation_summaries": [
                {
                    "title": "Python debugging",
                    "domains": ["Python"],
                    "learning_behaviors": ["debugging"],
                    "evidence": [
                        {
                            "evidence_id": "E001",
                            "text": "My Python code gives an import error. How do I fix it?",
                            "kind": "attempting_or_debugging, asking_for_help_or_explanation",
                            "source_note": "short_user_turn",
                        },
                        {
                            "evidence_id": "E002",
                            "text": "I tried changing the import and still get a traceback.",
                            "kind": "attempting_or_debugging",
                            "source_note": "short_user_turn",
                        },
                    ],
                }
            ],
        },
    }
    path = tmp_path / "rich_learner_context.json"
    path.write_text(json.dumps(compact), encoding="utf-8")

    class FakeCompactLLM:
        def generate(self, prompt):
            return json.dumps({
                "observed_patterns": [
                    {
                        "pattern": "Returns with continued debugging questions",
                        "observation": "Two messages describe trying a fix and asking about the remaining error.",
                        "evidence_ids": ["E001", "E002"],
                    }
                ],
                "explicit_preferences": [],
            })

    profile = build_profile(path, FakeCompactLLM())
    assert profile["topic_activity"] == [
        {"topic": "Python", "conversation_count": 1, "evidence_ids": ["E001", "E002"]},
        {"topic": "SQL", "conversation_count": 1, "evidence_ids": []},
    ]
    assert profile["observed_patterns"][0]["evidence_ids"] == ["E001", "E002"]
    assert profile["evidence"][0]["conversation_title"] == "Python debugging"
    assert "skill or mastery" in profile["not_inferable_from_chat_history"][0]


def test_groq_provider_uses_configured_model_and_prompt(monkeypatch):
    from types import SimpleNamespace
    import openai

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    provider = profile_builder.LLMProvider()
    assert provider.generate("test prompt") == '{"ok": true}'
    assert captured["client_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["request"]["model"] == "test-model"
    assert captured["request"]["messages"] == [
        {"role": "user", "content": "test prompt"}
    ]
    assert captured["request"]["response_format"] == {"type": "json_object"}


def test_groq_provider_discovers_active_model(monkeypatch):
    from types import SimpleNamespace
    import openai

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(data=[
                    SimpleNamespace(id="whisper-large-v3", active=True),
                    SimpleNamespace(id="llama-4-maverick-17b", active=True),
                    SimpleNamespace(id="llama-inactive", active=False),
                ])
            )

    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    provider = profile_builder.LLMProvider()
    assert provider.model_name == "llama-4-maverick-17b"
