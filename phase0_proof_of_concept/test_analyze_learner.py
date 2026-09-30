from analyze_learner import (
    calculate_message_signal,
    detect_domains,
    summarize_conversation,
)


def test_pasted_financial_aid_instructions_are_not_learning_evidence():
    text = (
        "If these benefits can be used only at certain schools, explain in the "
        "Special Circumstance section at the end of the application."
    )
    assert calculate_message_signal(text) == 0


def test_python_join_question_is_not_mislabeled_as_sql():
    text = "In Python, str.join() keeps giving me trouble. How does it work?"
    domains = detect_domains(text)
    assert "Python" in domains
    assert "SQL" not in domains
    assert calculate_message_signal(text) > 0


def test_java_evidence_keeps_java_domain():
    message = {
        "conversation_title": "Java file structure",
        "content": "System.out.println(obj.x); Why is this not allowed in Java?",
    }
    summary = summarize_conversation([message])
    assert "Java" in summary["domains"]