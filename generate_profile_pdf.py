from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

import json
from xml.sax.saxutils import escape

# ---------------- CONFIG ----------------

INPUT_JSON = "learner_profile.json"      # put your JSON here
OUTPUT_PDF = "learning_profile_full.pdf"  # output PDF

# ---------------- HELPERS ----------------

def load_profile(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)

def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='TitleCustom',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#17365D'),
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='HeadingCustom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1F4E79'),
        spaceBefore=12,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='BodyCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.2,
        leading=14,
        spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name='SmallCustom',
        parent=styles['BodyText'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#555555')
    ))
    return styles

def para(text: str, style):
    safe = escape(text or "")
    safe = safe.replace("\n", "<br/>")
    return Paragraph(safe, style)

# ---------------- CONTENT BUILDERS ----------------

def build_intro(profile: dict, styles) -> list:
    ds = profile.get("dataset_summary", {})
    total_msgs = ds.get("total_messages", "?")
    total_conv = ds.get("total_conversations", "?")
    sel_conv = ds.get("selected_conversations", "?")
    sel_ev = ds.get("selected_evidence_items", "?")

    story = []
    story.append(para("Learning Profile (Full Extraction)", styles['TitleCustom']))
    story.append(para(
        f"Plain-language summary based on the provided chat-history sample "
        f"({sel_conv} selected conversations, {sel_ev} evidence items, out of {total_msgs} messages "
        f"across {total_conv} conversations). This is not a formal skills assessment.",
        styles['SmallCustom']
    ))
    story.append(Spacer(1, 8))
    return story

def build_plain_summary(profile: dict, styles) -> list:
    story = []
    story.append(para("In simple terms", styles['HeadingCustom']))

    topics = [t.get("topic", "Unknown") for t in profile.get("topic_activity", [])]
    patterns = profile.get("observed_patterns", [])
    topic_text = ", ".join(topics) if topics else "the topics represented in the selected messages"
    pattern_text = (
        f"The evidence supports {len(patterns)} recurring interaction pattern(s). "
        if patterns
        else "No repeated interaction pattern was supported by this sample. "
    )
    summary = (
        f"The selected chat history includes activity across {topic_text}. "
        f"{pattern_text}This report describes messages brought to an AI assistant; "
        "it does not verify independent skill, mastery, or learning progress."
    )

    story.append(para(summary, styles['BodyCustom']))
    story.append(Spacer(1, 6))
    return story

def build_preferences_table(profile: dict, styles) -> list:
    story = []
    story.append(para("Explicit learning preferences", styles['HeadingCustom']))

    preferences = profile.get("explicit_preferences", [])
    rows = []
    for preference in preferences:
        title = preference.get("preference") or "Preference"
        details = preference.get("observation") or "Explicitly stated in the cited message."
        evidence_ids = ", ".join(preference.get("evidence_ids", []))
        if evidence_ids:
            details += f"\nEvidence: {evidence_ids}"
        rows.append((title, details))
    if not rows:
        rows.append((
            "No explicit preference identified",
            "The selected messages did not provide a directly stated learning preference. "
            "This does not mean the learner has no preferences.",
        ))

    data = [[
        Paragraph("<b>Observed preference</b>", styles['BodyCustom']),
        Paragraph("<b>What it means</b>", styles['BodyCustom'])
    ]]
    for a, b in rows:
        data.append([
            para(a, styles['BodyCustom']),
            para(b, styles['BodyCustom'])
        ])

    t = Table(data, colWidths=[5.0*cm, 11.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D9EAF7')),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#A6A6A6')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))

    story.append(t)
    story.append(Spacer(1, 8))
    return story

def build_topics(profile: dict, styles) -> list:
    story = []
    story.append(para("What they have been learning", styles['HeadingCustom']))

    topic_lines = []
    for t in profile.get("topic_activity", []):
        name = t.get("topic", "Unknown")
        conv_count = t.get("conversation_count", 0)
        ev_ids = t.get("evidence_ids", [])
        ev_str = ", ".join(str(e) for e in ev_ids) if ev_ids else "none"
        topic_lines.append(
            Paragraph(
                f"<b>{escape(str(name))}:</b> {escape(str(conv_count))} conversation(s); "
                f"evidence: {escape(ev_str)}.",
                styles['BodyCustom'],
            )
        )

    if not topic_lines:
        topic_lines = ["No topic activity recorded in this sample."]

    for line in topic_lines:
        if isinstance(line, Paragraph):
            story.append(line)
        else:
            story.append(para("• " + line, styles['BodyCustom']))

    story.append(Spacer(1, 6))
    return story

def build_patterns(profile: dict, styles) -> list:
    story = []
    story.append(para("Observed interaction patterns", styles['HeadingCustom']))

    patterns = profile.get("observed_patterns", [])
    if not patterns:
        story.append(para("No repeated interaction pattern was supported by the selected evidence.", styles['BodyCustom']))
    else:
        for p in patterns:
            title = p.get("pattern") or "Pattern"
            obs = p.get("observation") or ""
            ev_ids = p.get("evidence_ids", [])
            ev_str = ", ".join(str(e) for e in ev_ids) if ev_ids else "none"

            story.append(Paragraph(f"<b>{escape(str(title))}</b>", styles['BodyCustom']))
            story.append(para(obs, styles['BodyCustom']))
            story.append(para(f"Evidence: {ev_str}", styles['SmallCustom']))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 6))
    return story

def build_limitations(profile: dict, styles) -> list:
    story = []
    story.append(para("What this history cannot establish", styles['HeadingCustom']))

    limits = profile.get("not_inferable_from_chat_history", [])
    if limits:
        for lim in limits:
            story.append(para("• " + lim, styles['BodyCustom']))
    else:
        story.append(para(
            "This report does not independently verify skill, mastery, or learning progress.",
            styles['BodyCustom']
        ))

    story.append(Spacer(1, 8))
    return story

def build_evidence_appendix(profile: dict, styles) -> list:
    story = []
    story.append(PageBreak())
    story.append(para("Evidence appendix", styles['TitleCustom']))
    story.append(para(
        "Excerpts below are the source messages cited in the profile. Long or code-heavy messages may include pasted material.",
        styles['SmallCustom']
    ))
    story.append(Spacer(1, 8))

    evidence = profile.get("evidence", [])
    if not evidence:
        story.append(para("No evidence items were provided in this sample.", styles['BodyCustom']))
        return story

    # Simple list of evidence items
    for ev in evidence:
        ev_id = ev.get("evidence_id", "")
        conv_title = ev.get("conversation_title", "Untitled")
        source_note = ev.get("source_note", "")
        text = ev.get("text", "")
        domains = ev.get("domains", [])
        kind = ev.get("kind", "")

        header = f"<b>{escape(str(ev_id))}</b> — {escape(str(conv_title))}"
        if source_note:
            header += f" ({escape(str(source_note))})"
        if domains:
            header += " | " + escape(", ".join(domains))
        if kind:
            header += " | " + escape(str(kind))

        story.append(Paragraph(header, styles['HeadingCustom']))
        story.append(para(text, styles['BodyCustom']))
        story.append(Spacer(1, 8))

    return story

def build_pdf(profile: dict, output_path: Path):
    styles = make_styles()
    story = []

    story.extend(build_intro(profile, styles))
    story.extend(build_plain_summary(profile, styles))
    story.extend(build_preferences_table(profile, styles))
    story.extend(build_topics(profile, styles))
    story.extend(build_patterns(profile, styles))
    story.extend(build_limitations(profile, styles))
    story.extend(build_evidence_appendix(profile, styles))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8*cm,
        leftMargin=1.8*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm,
        title="Learning Profile (Full Extraction)",
        author="Learner AI Tutor"
    )
    doc.build(story)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_PDF)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    profile = load_profile(input_path)
    build_pdf(profile, output_path)
    print(f"PDF saved to: {output_path.resolve()}")