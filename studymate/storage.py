import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import TypedDict
from uuid import uuid4

from studymate.data import SUBJECTS, STUDENT, WEAK_TOPICS


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATABASE_PATH = DATA_DIR / "studymate.sqlite3"


class StudentProfile(TypedDict):
    first_name: str
    program: str
    term: str
    learning_preferences: str
    advisor_memory: str


class SubjectProgress(TypedDict):
    name: str
    progress: int
    next_step: str
    weekly_hours: int


class FocusTopic(TypedDict):
    id: int
    name: str
    subject: str


class StudyPlanSession(TypedDict):
    id: int
    study_date: str
    subject: str
    task: str
    duration_minutes: int
    status: str


class PlannedSession(TypedDict):
    study_date: str
    subject: str
    task: str
    duration_minutes: int


@contextmanager
def _connect(db_path: Path | str = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(db_path: Path | str = DATABASE_PATH) -> None:
    """Create the local profile/progress store and seed it once with demo data."""
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                first_name TEXT NOT NULL,
                program TEXT NOT NULL,
                term TEXT NOT NULL,
                learning_preferences TEXT NOT NULL DEFAULT '',
                advisor_memory TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_progress (
                name TEXT PRIMARY KEY,
                progress INTEGER NOT NULL CHECK (progress BETWEEN 0 AND 100),
                next_step TEXT NOT NULL,
                weekly_hours INTEGER NOT NULL CHECK (weekly_hours BETWEEN 0 AND 80)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS focus_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                UNIQUE (name, subject)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS study_plans (
                id TEXT PRIMARY KEY,
                approved_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS study_plan_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL REFERENCES study_plans(id) ON DELETE CASCADE,
                study_date TEXT NOT NULL,
                subject TEXT NOT NULL,
                task TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL CHECK (duration_minutes BETWEEN 15 AND 240),
                status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'completed'))
            )
            """
        )

        connection.execute(
            """
            INSERT OR IGNORE INTO student_profile
                (id, first_name, program, term)
            VALUES (1, ?, ?, ?)
            """,
            (STUDENT["first_name"], STUDENT["program"], STUDENT["term"]),
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO subject_progress (name, progress, next_step, weekly_hours)
            VALUES (:name, :progress, :next_step, :weekly_hours)
            """,
            SUBJECTS,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO focus_topics (name, subject) VALUES (:name, :subject)",
            WEAK_TOPICS,
        )


def get_student_profile(db_path: Path | str = DATABASE_PATH) -> StudentProfile:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT first_name, program, term, learning_preferences, advisor_memory
            FROM student_profile WHERE id = 1
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("The StudyMate profile store has not been initialized.")
    return {
        "first_name": row["first_name"],
        "program": row["program"],
        "term": row["term"],
        "learning_preferences": row["learning_preferences"],
        "advisor_memory": row["advisor_memory"],
    }


def save_student_profile(
    first_name: str,
    program: str,
    term: str,
    learning_preferences: str,
    advisor_memory: str,
    db_path: Path | str = DATABASE_PATH,
) -> None:
    values = {
        "first_name": first_name.strip(),
        "program": program.strip(),
        "term": term.strip(),
        "learning_preferences": learning_preferences.strip(),
        "advisor_memory": advisor_memory.strip(),
    }
    if not all(values[key] for key in ("first_name", "program", "term")):
        raise ValueError("Name, program, and term are required.")
    if len(values["first_name"]) > 80 or len(values["program"]) > 120 or len(values["term"]) > 80:
        raise ValueError("Name, program, or term exceeds the allowed length.")
    if len(values["learning_preferences"]) > 2_000 or len(values["advisor_memory"]) > 4_000:
        raise ValueError("Learning preferences or advisor memory is too long.")

    with _connect(db_path) as connection:
        connection.execute(
            """
            UPDATE student_profile
            SET first_name = :first_name,
                program = :program,
                term = :term,
                learning_preferences = :learning_preferences,
                advisor_memory = :advisor_memory
            WHERE id = 1
            """,
            values,
        )


def get_subject_progress(db_path: Path | str = DATABASE_PATH) -> list[SubjectProgress]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT name, progress, next_step, weekly_hours
            FROM subject_progress ORDER BY name
            """
        ).fetchall()
    return [
        {
            "name": row["name"],
            "progress": row["progress"],
            "next_step": row["next_step"],
            "weekly_hours": row["weekly_hours"],
        }
        for row in rows
    ]


def save_subject_progress(
    name: str,
    progress: int,
    next_step: str,
    weekly_hours: int,
    db_path: Path | str = DATABASE_PATH,
) -> None:
    if not 0 <= progress <= 100:
        raise ValueError("Subject progress must be between 0 and 100.")
    if not 0 <= weekly_hours <= 80:
        raise ValueError("Weekly study hours must be between 0 and 80.")
    if len(next_step.strip()) > 240:
        raise ValueError("The next step must be 240 characters or fewer.")

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE subject_progress
            SET progress = ?, next_step = ?, weekly_hours = ?
            WHERE name = ?
            """,
            (progress, next_step.strip(), weekly_hours, name),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected subject no longer exists.")


def get_focus_topics(db_path: Path | str = DATABASE_PATH) -> list[FocusTopic]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, name, subject FROM focus_topics ORDER BY subject, name"
        ).fetchall()
    return [
        {"id": row["id"], "name": row["name"], "subject": row["subject"]}
        for row in rows
    ]


def add_focus_topic(
    name: str,
    subject: str,
    db_path: Path | str = DATABASE_PATH,
) -> bool:
    clean_name = name.strip()
    clean_subject = subject.strip()
    if not clean_name or not clean_subject:
        raise ValueError("A topic and subject are required.")
    if len(clean_name) > 120 or len(clean_subject) > 120:
        raise ValueError("Topic and subject must each be 120 characters or fewer.")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO focus_topics (name, subject) VALUES (?, ?)",
            (clean_name, clean_subject),
        )
        return cursor.rowcount == 1


def delete_focus_topic(topic_id: int, db_path: Path | str = DATABASE_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM focus_topics WHERE id = ?", (topic_id,))


def approve_study_plan(
    sessions: list[PlannedSession],
    db_path: Path | str = DATABASE_PATH,
) -> str:
    """Persist a user-approved immutable plan and make it the active plan."""
    if not sessions:
        raise ValueError("Cannot approve an empty study plan.")
    for session in sessions:
        try:
            date.fromisoformat(session["study_date"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each study session needs a valid ISO date.") from error
        if not session["subject"].strip() or not session["task"].strip():
            raise ValueError("Each study session needs a subject and task.")
        if not 15 <= session["duration_minutes"] <= 240:
            raise ValueError("Study sessions must be between 15 and 240 minutes.")

    plan_id = str(uuid4())
    approved_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as connection:
        connection.execute("UPDATE study_plans SET is_active = 0 WHERE is_active = 1")
        connection.execute(
            "INSERT INTO study_plans (id, approved_at, is_active) VALUES (?, ?, 1)",
            (plan_id, approved_at),
        )
        connection.executemany(
            """
            INSERT INTO study_plan_sessions
                (plan_id, study_date, subject, task, duration_minutes)
            VALUES (:plan_id, :study_date, :subject, :task, :duration_minutes)
            """,
            [{"plan_id": plan_id, **session} for session in sessions],
        )
    return plan_id


def get_active_study_plan(
    db_path: Path | str = DATABASE_PATH,
) -> list[StudyPlanSession]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT session.id, session.study_date, session.subject, session.task,
                   session.duration_minutes, session.status
            FROM study_plan_sessions AS session
            JOIN study_plans AS plan ON plan.id = session.plan_id
            WHERE plan.is_active = 1
            ORDER BY session.study_date, session.id
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "study_date": row["study_date"],
            "subject": row["subject"],
            "task": row["task"],
            "duration_minutes": row["duration_minutes"],
            "status": row["status"],
        }
        for row in rows
    ]


def set_study_session_status(
    session_id: int,
    status: str,
    db_path: Path | str = DATABASE_PATH,
) -> None:
    if status not in {"planned", "completed"}:
        raise ValueError("Session status must be planned or completed.")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE study_plan_sessions
            SET status = ?
            WHERE id = ? AND plan_id IN (
                SELECT id FROM study_plans WHERE is_active = 1
            )
            """,
            (status, session_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("The active study session no longer exists.")
