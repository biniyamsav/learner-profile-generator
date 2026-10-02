from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import json

# ---------------- CONFIG ----------------

INPUT_JSON = "learner_profile.json"
OUTPUT_PDF = "learning_profile_full.pdf"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#1F4E79")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GRAY = colors.HexColor("#F4F6F8")
MID_GRAY = colors.HexColor("#64748B")
GRID = colors.HexColor("#D7E0E8")
CONTENT_WIDTH = 17.4 * cm

# ---------------- HELPERS ----------------


def load_profile(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ProfileTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProfileSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MID_GRAY,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#243447"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=MID_GRAY,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardNumber",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=NAVY,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=NAVY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#243447"),
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EvidenceTitle",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=NAVY,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EvidenceText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#243447"),
            spaceAfter=0,
        )
    )
    return styles


def para(text: str, style):
    safe = escape(str(text or ""))
    safe = safe.replace("\n", "<br/>")
    return Paragraph(safe, style)


def _profile_evidence(profile: dict) -> dict[str, dict]:
    return {
        str(item["evidence_id"]): item
        for item in profile.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }


def _display_date(timestamp) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp), timezone.utc).strftime("%b %Y")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(doc.leftMargin, 0.9 * cm, "Learner activity profile")
    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        0.9 * cm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _section_heading(title: str, styles) -> Paragraph:
    return para(title, styles["SectionHeading"])


def _metric_card(value, label, styles) -> Table:
    card = Table(
        [[para(value, styles["CardNumber"])], [para(label, styles["CardLabel"])]],
        colWidths=[CONTENT_WIDTH / 4 - 4],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
            ]
        )
    )
    return card


# ---------------- CONTENT BUILDERS ----------------


def build_intro(profile: dict, styles) -> list:
    ds = profile.get("dataset_summary", {})
    selected_conversations = ds.get("selected_conversations", 0)
    selected_evidence = ds.get("selected_evidence_items", len(profile.get("evidence", [])))
    total_messages = ds.get("total_messages", 0)
    total_conversations = ds.get("total_conversations", 0)

    story = [
        para("Learning Profile", styles["ProfileTitle"]),
        para(
            "A clear, evidence-linked summary of activity in the selected conversation sample. "
            "It describes what appears in the messages; it is not a formal assessment of ability.",
            styles["ProfileSubtitle"],
        ),
    ]

    metrics = [
        _metric_card(selected_conversations, "CONVERSATIONS IN SAMPLE", styles),
        _metric_card(selected_evidence, "EVIDENCE ITEMS", styles),
        _metric_card(total_conversations, "CONVERSATIONS IN HISTORY", styles),
        _metric_card(total_messages, "MESSAGES IN HISTORY", styles),
    ]
    cards = Table([metrics], colWidths=[CONTENT_WIDTH / 4] * 4, hAlign="LEFT")
    cards.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.extend([cards, Spacer(1, 5)])

    if selected_conversations and total_conversations:
        story.append(
            para(
                f"The report uses {selected_conversations} selected conversations "
                f"from a history of {total_conversations}. The selected messages are a sample, "
                "so counts describe this report's evidence, not necessarily the full history.",
                styles["SmallCustom"],
            )
        )
    return story


def build_plain_summary(profile: dict, styles) -> list:
    topics = profile.get("topic_activity", [])
    patterns = profile.get("observed_patterns", [])
    topic_names = [str(item.get("topic", "Unknown")) for item in topics if isinstance(item, dict)]
    if topic_names:
        topic_text = ", ".join(topic_names)
        summary = (
            f"The selected messages cover {len(topic_names)} recorded topic areas: {topic_text}. "
        )
    else:
        summary = "No topic areas were recorded for the selected messages. "

    if patterns:
        summary += (
            f"The profile also records {len(patterns)} recurring observation(s), "
            "each linked to cited messages below. "
        )
    else:
        summary += "No recurring observation was supported in the selected evidence. "
    summary += (
        "A request for help or pasted code is evidence of activity, not proof of independent "
        "skill, mastery, or learning progress."
    )
    return [_section_heading("In brief", styles), para(summary, styles["BodyCustom"])]


def build_topics(profile: dict, styles) -> list:
    story = [_section_heading("Topics represented in the selected history", styles)]
    topics = profile.get("topic_activity", [])
    if not topics:
        story.append(para("No topic activity was recorded in this sample.", styles["BodyCustom"]))
        return story

    rows = [[
        para("Topic", styles["TableHead"]),
        para("Conversations", styles["TableHead"]),
        para("Cited messages", styles["TableHead"]),
    ]]
    for item in topics:
        if not isinstance(item, dict):
            continue
        evidence_ids = item.get("evidence_ids", [])
        rows.append(
            [
                para(item.get("topic", "Unknown"), styles["TableBody"]),
                para(item.get("conversation_count", 0), styles["TableBody"]),
                para(len(set(map(str, evidence_ids))), styles["TableBody"]),
            ]
        )

    table = Table(rows, colWidths=[9.4 * cm, 4.0 * cm, 4.0 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([table, Spacer(1, 3)])
    return story


def build_message_types(profile: dict, styles) -> list:
    counts: dict[str, int] = {}
    for item in profile.get("evidence", []):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "")
        tags = {tag.strip() for tag in str(kind).split(",") if tag.strip()}
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1

    if not counts:
        return []

    story = [_section_heading("Kinds of messages in the evidence", styles)]
    rows = []
    for kind, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        rows.append(
            [
                para(kind.replace("_", " "), styles["TableBody"]),
                para(count, styles["TableBody"]),
            ]
        )
    table = Table(rows, colWidths=[13.4 * cm, 4.0 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.35, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            table,
            para(
                "A message can carry more than one kind, so these counts may overlap.",
                styles["SmallCustom"],
            ),
        ]
    )
    return story


def build_patterns(profile: dict, styles) -> list:
    story = [_section_heading("Recurring observations", styles)]
    evidence_by_id = _profile_evidence(profile)
    patterns = profile.get("observed_patterns", [])
    if not patterns:
        story.append(
            para(
                "No recurring observation was supported by the selected evidence.",
                styles["BodyCustom"],
            )
        )
        return story

    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        title = pattern.get("pattern") or "Observation"
        observation = pattern.get("observation") or ""
        evidence_ids = [str(item) for item in pattern.get("evidence_ids", [])]
        refs = ", ".join(evidence_ids) if evidence_ids else "No evidence IDs provided"
        story.append(Paragraph(f"<b>{escape(str(title))}</b>", styles["BodyCustom"]))
        if observation:
            story.append(para(observation, styles["BodyCustom"]))
        story.append(para(f"Sources: {refs}", styles["SmallCustom"]))

        cited_titles = []
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence:
                title_text = evidence.get("conversation_title")
                if title_text and title_text not in cited_titles:
                    cited_titles.append(title_text)
        if cited_titles:
            story.append(
                para(
                    "Related conversations: " + "; ".join(map(str, cited_titles)),
                    styles["SmallCustom"],
                )
            )
        story.append(Spacer(1, 3))
    return story


def build_preferences_table(profile: dict, styles) -> list:
    story = [_section_heading("Explicitly stated preferences", styles)]
    preferences = profile.get("explicit_preferences", [])
    evidence_by_id = _profile_evidence(profile)
    if not preferences:
        story.append(
            para(
                "No explicit learning preference was identified in this sample. "
                "That does not mean the learner has no preferences.",
                styles["BodyCustom"],
            )
        )
        return story

    rows = [[
        para("Preference", styles["TableHead"]),
        para("Source messages", styles["TableHead"]),
    ]]
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        evidence_ids = [str(item) for item in preference.get("evidence_ids", [])]
        sources = []
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence:
                title = evidence.get("conversation_title", "Untitled conversation")
                sources.append(f"{evidence_id} - {title}")
            else:
                sources.append(evidence_id)
        rows.append(
            [
                para(preference.get("preference") or "Preference", styles["TableBody"]),
                para("; ".join(sources) or "No cited source", styles["TableBody"]),
            ]
        )

    table = Table(rows, colWidths=[6.2 * cm, 11.2 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    return story


def build_limitations(profile: dict, styles) -> list:
    story = [_section_heading("How to interpret this profile", styles)]
    limits = profile.get("not_inferable_from_chat_history", [])
    if limits:
        for limit in limits:
            story.append(para(f"- {limit}", styles["BodyCustom"]))
    else:
        story.append(
            para(
                "Conversation history cannot independently verify skill, mastery, "
                "or learning progress.",
                styles["BodyCustom"],
            )
        )
    return story


def build_evidence_appendix(profile: dict, styles) -> list:
    evidence = profile.get("evidence", [])
    story = [
        para("Evidence record", styles["ProfileTitle"]),
        para(
            "Every selected evidence item is listed below. Sources and message text are "
            "included so the observations in the profile can be checked against the original sample.",
            styles["ProfileSubtitle"],
        ),
    ]
    if not evidence:
        story.append(para("No evidence items were provided in this sample.", styles["BodyCustom"]))
        return story

    rows = [[
        para("ID and source", styles["TableHead"]),
        para("Selected message text", styles["TableHead"]),
    ]]
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id", "")
        title = item.get("conversation_title", "Untitled conversation")
        date = _display_date(item.get("create_time"))
        domains = item.get("domains", [])
        kind = item.get("kind", "")
        source_note = item.get("source_note", "")

        metadata = [f"<b>{escape(str(evidence_id))}</b>"]
        metadata.append(escape(str(title)))
        details = [str(value) for value in (date, ", ".join(map(str, domains))) if value]
        if details:
            metadata.append(escape(" | ".join(details)))
        if kind:
            metadata.append(escape(str(kind).replace("_", " ")))
        if source_note:
            metadata.append(escape(str(source_note).replace("_", " ")))
        if item.get("message_id"):
            metadata.append(f"Message ID: {escape(str(item['message_id']))}")

        rows.append(
            [
                Paragraph("<br/>".join(metadata), styles["SmallCustom"]),
                para(item.get("text", ""), styles["EvidenceText"]),
            ]
        )

    table = LongTable(
        rows,
        colWidths=[5.0 * cm, CONTENT_WIDTH - 5.0 * cm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    return story


def build_pdf(profile: dict, output_path: Path):
    styles = make_styles()
    story = []

    story.extend(build_intro(profile, styles))
    story.extend(build_plain_summary(profile, styles))
    story.extend(build_topics(profile, styles))
    story.extend(build_message_types(profile, styles))
    story.extend(build_patterns(profile, styles))
    story.extend(build_preferences_table(profile, styles))
    story.extend(build_limitations(profile, styles))
    story.extend(build_evidence_appendix(profile, styles))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.8 * cm,
        title="Learner Activity Profile",
        author="Learner AI Tutor",
    )
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_PDF)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    profile = load_profile(input_path)
    build_pdf(profile, output_path)
    print(f"PDF saved to: {output_path.resolve()}")
