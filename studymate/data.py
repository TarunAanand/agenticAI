from datetime import date, timedelta


STUDENT = {
    "first_name": "Alex",
    "program": "Computer Science",
    "term": "Fall 2026",
    "completed_tasks": 18,
    "total_tasks": 24,
    "study_streak": 5,
}

SUBJECTS = [
    {
        "name": "Data Structures",
        "progress": 78,
        "next_step": "Review graph traversal",
        "weekly_hours": 6,
    },
    {
        "name": "Linear Algebra",
        "progress": 56,
        "next_step": "Practice eigenvalue problems",
        "weekly_hours": 4,
    },
    {
        "name": "Database Systems",
        "progress": 84,
        "next_step": "Finish normalization notes",
        "weekly_hours": 5,
    },
]

DEADLINES = [
    {
        "title": "Graph Algorithms Problem Set",
        "subject": "Data Structures",
        "due": "Oct 8",
        "due_date": "2026-10-08",
        "kind": "Assignment",
    },
    {
        "title": "Eigenvalues Quiz",
        "subject": "Linear Algebra",
        "due": "Oct 11",
        "due_date": "2026-10-11",
        "kind": "Quiz",
    },
    {
        "title": "Database Design Project",
        "subject": "Database Systems",
        "due": "Oct 16",
        "due_date": "2026-10-16",
        "kind": "Project",
    },
]

WEAK_TOPICS = [
    {"name": "Eigenvalues and eigenvectors", "subject": "Linear Algebra"},
    {"name": "Dijkstra's algorithm", "subject": "Data Structures"},
    {"name": "Normalization", "subject": "Database Systems"},
]


def get_upcoming_deadlines(
    as_of: date | None = None,
    days: int = 14,
) -> list[dict[str, str]]:
    """Return sample deadlines from today through the requested horizon."""
    if days < 0:
        raise ValueError("Deadline horizon cannot be negative.")
    today = as_of or date.today()
    last_day = today + timedelta(days=days)
    upcoming = []
    for deadline in DEADLINES:
        due_date = date.fromisoformat(deadline["due_date"])
        if today <= due_date <= last_day:
            day_label = due_date.strftime("%b %d").replace(" 0", " ")
            upcoming.append({**deadline, "due": day_label})
    return sorted(upcoming, key=lambda deadline: deadline["due_date"])
