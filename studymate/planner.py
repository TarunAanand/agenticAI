from datetime import date, timedelta

from studymate.data import DEADLINES
from studymate.storage import FocusTopic, PlannedSession, SubjectProgress


def build_weekly_draft(
    subjects: list[SubjectProgress],
    focus_topics: list[FocusTopic],
    study_days: list[int],
    minutes_per_day: int,
    start_date: date | None = None,
) -> list[PlannedSession]:
    """Draft local study sessions, prioritizing near deadlines and weaker subjects."""
    if not study_days:
        raise ValueError("Choose at least one study day.")
    if any(day < 0 or day > 6 for day in study_days):
        raise ValueError("Study days must be weekdays numbered 0 through 6.")
    if not 25 <= minutes_per_day <= 240:
        raise ValueError("Daily study time must be between 25 and 240 minutes.")
    if not subjects:
        return []

    today = start_date or date.today()
    deadline_by_subject = {
        item["subject"]: date.fromisoformat(item["due_date"])
        for item in DEADLINES
        if date.fromisoformat(item["due_date"]) >= today
    }
    ordered_subjects = sorted(
        subjects,
        key=lambda item: (
            deadline_by_subject.get(item["name"], date.max),
            item["progress"],
            item["name"],
        ),
    )
    recommendations: list[tuple[str, str]] = [
        (subject["name"], subject["next_step"] or "Review recent course material")
        for subject in ordered_subjects
    ]
    topic_groups: dict[str, list[str]] = {}
    for topic in focus_topics:
        topic_groups.setdefault(topic["subject"], []).append(topic["name"])
    for subject in ordered_subjects:
        for topic in sorted(topic_groups.get(subject["name"], [])):
            recommendations.append((subject["name"], f"Practice: {topic}"))
    if not recommendations:
        return []

    selected_days = set(study_days)
    dates = [
        today + timedelta(days=offset)
        for offset in range(1, 8)
        if (today + timedelta(days=offset)).weekday() in selected_days
    ]
    if not dates:
        return []

    sessions: list[PlannedSession] = []
    candidate_index = 0
    for study_date in dates:
        remaining = minutes_per_day
        while remaining >= 25:
            duration = min(45, remaining)
            subject, task = recommendations[candidate_index % len(recommendations)]
            sessions.append(
                {
                    "study_date": study_date.isoformat(),
                    "subject": subject,
                    "task": task,
                    "duration_minutes": duration,
                }
            )
            candidate_index += 1
            remaining -= duration
    return sessions
