import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

INPUT_FILE = Path("user_messages.json")
OUTPUT_FILE = Path("rich_learner_context.json")

MAX_CONVERSATIONS = 16
MAX_EVIDENCE_PER_CONVO = 2
MAX_SNIPPET_CHARS = 260

LEARNING_REQUEST_RE = re.compile(
    r"\b(can you|could you|please|explain|help me|teach me|show me|fix|debug|review|"
    r"i want to understand|i don't understand|what does this mean|how do i|how does|"
    r"why does|solve|calculate|derive|compare|which of these|what is the difference)\b",
    re.IGNORECASE,
)
LEARNER_ATTEMPT_RE = re.compile(
    r"\b(i tried|i wrote|i made|i got|i expected|my code|my program|it gives|"
    r"it returns|i am stuck|i'm stuck|i don't get|still doesn't work)\b",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"\b(traceback|exception|error|failed|not working|doesn't work|undefined|"
    r"attributeerror|typeerror|valueerror|nameerror)\b",
    re.IGNORECASE,
)

TOPIC_KEYWORDS = {
    "Python": ["python", "pandas", "pd.", "numpy", "streamlit", "plotly", "altair", "flask", "sklearn", "scikit-learn", "matplotlib", "dataframe", "jupyter"],
    "SQL": ["sql", "postgres", "postgresql", "mysql", "sqlite", "psycopg", "sqlalchemy", "database schema"],
    "C++": ["#include", "iostream", "using namespace std", "vector", "template<typename", "c++", "std::", "g++", "int main"],
    "Java": ["java", "jvm", "public static void main", "system.out.println", "javac"],
    "Data Science": ["data science", "machine learning", "classification", "regression", "accuracy", "training data", "validation set", "feature engineering", "dataset", "tensorflow", "pytorch", "overfitting", "underfitting"],
    "Web App": ["streamlit", "flask", "html", "css", "api", "frontend", "backend", "dashboard", "request.get_json", "app.route", "endpoint"],
    "Big Data": ["kafka", "spark", "pyspark", "hdfs", "trino", "distributed computing", "partition", "broker"],
    "Math/Stats": ["derivative", "probability", "probability distribution", "binomial", "expectation", "conditional probability", "log-normal", "integral", "standard deviation", "hypothesis test", "combinatorics"],
    "Linux/Git": ["git", "bash", "zsh", "powershell", "terminal", "venv", "pip install", "git status", "ubuntu", "linux", "command line"],
    "Blockchain": ["blockchain", "ethereum", "metamask", "wallet", "smart contract", "tenderly", "gas fee", "transaction"],
}

LEARNING_BEHAVIORS = [
    ("project_building", ["project", "dashboard", "app", "build", "system", "analytics", "database", "web app"]),
    ("debugging", ["traceback", "error", "attributeerror", "valueerror", "nameerror", "typeerror", "not defined", "failed", "bug", "exception"]),
    ("concept_learning", ["what is", "how does", "explain", "elaborate", "what do you mean", "tell me", "why"]),
    ("exam_or_quiz", ["question", "quiz", "test", "mcq", "exam", "practice", "what is the answer", "which one"]),
    ("environment_setup", ["install", "pip", "venv", "requirements", "conda", "python version", "gcc", "dependency"]),
    ("answer_seeking", ["give me all the answer", "answer", "solve", "do this for me", "full valid one"]),
]

SPECIAL_TERMS = {
    "python": ["pandas", "numpy", "streamlit", "plotly", "altair", "flask", "sklearn", "matplotlib"],
    "sql": ["select", "join", "where", "database", "postgres", "mysql"],
    "cpp": ["#include", "using namespace std", "vector", "template<typename", "cout", "cin"],
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_message_signal(text: str) -> float:
    """Score topic-specific learner questions and attempts, not pasted text volume."""
    text = normalize_text(text)
    if not text or len(text) < 6:
        return 0.0

    if not detect_domains(text):
        return 0.0

    recent_text = text[-300:]
    has_request = bool(LEARNING_REQUEST_RE.search(recent_text))
    has_attempt = bool(LEARNER_ATTEMPT_RE.search(recent_text))
    has_error = bool(ERROR_RE.search(recent_text))
    has_question = "?" in recent_text
    if not (has_request or has_attempt or has_error or has_question):
        return 0.0

    score = 2.0 + min(recent_text.count("?"), 2)
    score += 2.0 if has_request else 0.0
    score += 2.0 if has_attempt else 0.0
    score += 1.0 if has_error else 0.0

    if len(text) > 1200:
        score -= 2.0
    code_chars = len(re.findall(r"[{}\[\];=#]", text))
    if code_chars / max(len(text), 1) > 0.08:
        score -= 1.0

    return max(score, 0.0)


def detect_domains(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for domain, keywords in TOPIC_KEYWORDS.items():
        if any(
            keyword.lower() in lower
            if " " in keyword or any(char in keyword for char in "#+._")
            else re.search(rf"\b{re.escape(keyword.lower())}\b", lower)
            for keyword in keywords
        ):
            hits.append(domain)
    if re.search(r"\bselect\b.{0,160}\bfrom\b|\bjoin\b.{0,100}\bon\b", lower):
        hits.append("SQL")
    return hits[:3]


def detect_learning_behaviors(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    if LEARNER_ATTEMPT_RE.search(text) or ERROR_RE.search(text):
        hits.append("attempting_or_debugging")
    if LEARNING_REQUEST_RE.search(text) or "?" in text:
        hits.append("asking_for_help_or_explanation")
    if re.search(r"\b(build|create|implement|develop)\b.{0,80}\b(app|project|system|program|dashboard)\b", lower):
        hits.append("project_work")
    if re.search(r"\b(exam|quiz|mcq|practice questions?)\b", lower):
        hits.append("study_or_assessment")
    return hits[:3]


def classify_evidence_source(text: str) -> str:
    """Flag long/code-heavy user turns as possibly containing pasted material."""
    text = normalize_text(text)
    code_chars = len(re.findall(r"[{}\[\];=#]", text))
    if len(text) > 1200 or code_chars / max(len(text), 1) > 0.08:
        return "possibly_pasted_or_code_heavy"
    return "short_user_turn"


def build_evidence_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    cue_matches = list(
        re.finditer(
            r"\?|\b(can you|could you|please|explain|help me|teach me|show me|fix|debug|"
            r"review|i tried|i wrote|i got|my code|it gives|it returns|i don't understand)\b",
            text,
            re.IGNORECASE,
        )
    )
    if cue_matches:
        cue = cue_matches[-1]
        start = max(0, cue.start() - 90)
        end = min(len(text), start + max_chars)
        excerpt = text[start:end].strip()
        return ("..." if start else "") + excerpt + ("..." if end < len(text) else "")

    return text[:max_chars].rstrip(" .,:;-") + "..."


def summarize_conversation(messages: list[dict[str, Any]]) -> dict[str, Any]:
    clean_messages = [normalize_text(m.get("content", "")) for m in messages if normalize_text(m.get("content", ""))]
    if not clean_messages:
        return {}

    seen_messages: set[str] = set()
    scored = []
    for message in messages:
        text = normalize_text(message.get("content", ""))
        if not text or text in seen_messages:
            continue
        seen_messages.add(text)
        signal = calculate_message_signal(text)
        if signal > 0:
            scored.append((signal, text, message))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:MAX_EVIDENCE_PER_CONVO]
    if not selected:
        return {}

    evidence = [
        {
            "text": build_evidence_snippet(text),
            "kind": ", ".join(detect_learning_behaviors(text)) or "learner_question_or_request",
            "source_note": classify_evidence_source(text),
            "message_id": str(message.get("message_id", "")),
            "create_time": message.get("create_time"),
        }
        for _, text, message in selected
    ]

    domains = []
    for item in evidence:
        domains.extend(detect_domains(item["text"]))
    domains = sorted({d: 0 for d in domains}.keys(), key=lambda d: (-domains.count(d), d))

    behaviors = []
    for item in evidence:
        behaviors.extend(item["kind"].split(", "))
    behaviors = sorted({b: 0 for b in behaviors}.keys(), key=lambda b: (-behaviors.count(b), b))

    top_domains = []
    for domain in ["Python", "SQL", "Data Science", "Web App", "C++", "Java", "Big Data", "Math/Stats", "Linux/Git", "Blockchain"]:
        if domain in domains:
            top_domains.append(domain)

    summary = {
        "title": messages[0].get("conversation_title", "Untitled"),
        "domains": top_domains[:3] or ["General Learning"],
        "learning_behaviors": behaviors[:3] or ["concept_learning"],
        "message_count": len(messages),
        "evidence": evidence,
    }
    return summary


def build_larger_context(json_path: Path):
    if not json_path.exists():
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in raw_data:
        cid = str(msg.get("conversation_id", "unknown"))
        content = normalize_text(msg.get("content", ""))
        if content:
            conversations[cid].append(msg)

    scored_conversations: list[tuple[float, str, list[dict[str, Any]]]] = []
    for cid, msgs in conversations.items():
        message_scores = sorted(
            (calculate_message_signal(m.get("content", "")) for m in msgs),
            reverse=True,
        )
        score = sum(message_scores[:4])
        title = msgs[0].get("conversation_title", "Untitled")
        if score > 0:
            scored_conversations.append((score, title, msgs))

    scored_conversations.sort(key=lambda x: x[0], reverse=True)
    selected = scored_conversations[:MAX_CONVERSATIONS]

    summaries = []
    for _, title, msgs in selected:
        convo = summarize_conversation(msgs)
        if convo:
            summaries.append(convo)

    evidence_number = 1
    for convo in summaries:
        for item in convo["evidence"]:
            item["evidence_id"] = f"E{evidence_number:03d}"
            evidence_number += 1

    domain_frequency: dict[str, int] = defaultdict(int)
    behavior_frequency: dict[str, int] = defaultdict(int)
    for convo in summaries:
        for d in convo.get("domains", []):
            domain_frequency[d] += 1
        for b in convo.get("learning_behaviors", []):
            behavior_frequency[b] += 1

    primary_domains = sorted(domain_frequency.items(), key=lambda x: (-x[1], x[0]))[:5]
    primary_behaviors = sorted(behavior_frequency.items(), key=lambda x: (-x[1], x[0]))[:5]

    final_payload = {
        "dataset_summary": {
            "total_messages": len(raw_data),
            "total_conversations": len(conversations),
            "selected_conversations": len(summaries),
            "selected_evidence_items": evidence_number - 1,
            "estimated_prompt_tokens": max(300, int(sum(len(json.dumps(s, ensure_ascii=False)) for s in summaries) / 4)),
        },
        "learner_profile_context": {
            "main_domains": [d for d, _ in primary_domains],
            "learning_behaviors": [b for b, _ in primary_behaviors],
            "domain_conversation_counts": {d: count for d, count in primary_domains},
            "profile_limits": [
                "This data shows what the user brought to ChatGPT, not independently verified skill or mastery.",
                "Long or code-heavy user turns may contain pasted material; treat them as context, not proof of the user's ability.",
                "Do not infer learning preferences unless the user explicitly states them or repeats a clear preference.",
            ],
            "conversation_summaries": summaries,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(raw_data):,} total messages from {json_path.name}")
    print(f"Selected {len(summaries)} high-signal conversations")
    print(f"Saved compact learner context to {OUTPUT_FILE}")
    print(f"Estimated size: ~{len(json.dumps(final_payload, ensure_ascii=False)):,} characters")


if __name__ == "__main__":
    build_larger_context(INPUT_FILE)