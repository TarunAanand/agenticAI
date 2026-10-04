import hashlib
import html
import html
from datetime import date
from uuid import uuid4

import streamlit as st

from studymate.config import groq_api_key_configured
from studymate.data import STUDENT, get_upcoming_deadlines
from studymate.groq_client import (
    ChatMessage,
    GroqAPIError,
)
from studymate.materials import (
    MAX_FILE_BYTES,
    MaterialError,
    extract_and_chunk,
    retrieve_chunks,
)
from studymate.planner import build_weekly_draft
from studymate.quiz import QuizQuestion, QuizResult, score_quiz
from studymate.storage import (
    add_focus_topic,
    approve_study_plan,
    delete_focus_topic,
    get_active_study_plan,
    get_focus_topics,
    get_student_profile,
    get_subject_progress,
    initialize_database,
    save_student_profile,
    save_subject_progress,
    set_study_session_status,
)
from studymate.workflows import run_advisor_workflow, run_quiz_workflow


def _material_state() -> tuple[list[dict], list[dict]]:
    documents = st.session_state.setdefault("materials_documents", [])
    chunks = st.session_state.setdefault("materials_chunks", [])
    return documents, chunks


def _next_material_upload_key() -> str:
    generation = st.session_state.get("materials_uploader_generation", 0)
    return f"materials_uploader_{generation}"


def _reset_material_uploader() -> None:
    st.session_state["materials_uploader_generation"] = (
        st.session_state.get("materials_uploader_generation", 0) + 1
    )


st.set_page_config(
    page_title="StudyMate AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #eef2ff; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(115deg, #4338ca, #6366f1 58%, #818cf8);
        margin-bottom: 1.25rem;
    }
    .hero h1 { color: white; margin: 0 0 .45rem; font-size: 2.1rem; }
    .hero p { color: #e0e7ff; margin: 0; font-size: 1.05rem; }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .74rem;
        font-weight: 700;
        color: #c7d2fe;
        margin-bottom: .6rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8eaf2;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    h2, h3 { color: #1f2937; }
    .muted { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_dashboard() -> None:
    profile = get_student_profile()
    subjects = get_subject_progress()
    focus_topics = get_focus_topics()
    upcoming_deadlines = get_upcoming_deadlines(days=14)
    display_name = html.escape(profile["first_name"])
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Your academic command center</div>
            <h1>Good to see you, {display_name}.</h1>
            <p>A little progress each day adds up. Here is your study snapshot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    average_progress = (
        round(sum(subject["progress"] for subject in subjects) / len(subjects))
        if subjects
        else 0
    )
    metrics = st.columns(4)
    metrics[0].metric("Average subject progress", f"{average_progress}%", profile["term"])
    metrics[1].metric("Study streak", f'{STUDENT["study_streak"]} days', "Keep it going")
    metrics[2].metric("Upcoming deadlines", str(len(upcoming_deadlines)), "Next 14 days")
    metrics[3].metric("Focus topics", str(len(focus_topics)), "Worth revisiting")

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Subject progress")
        for subject in subjects:
            with st.container(border=True):
                label, score = st.columns([4, 1])
                label.markdown(f"**{subject['name']}**")
                score.markdown(f"**{subject['progress']}%**")
                st.progress(subject["progress"] / 100)
                st.caption(f'{subject["next_step"]} · {subject["weekly_hours"]} hrs this week')

    with right:
        st.subheader("Coming up")
        if not upcoming_deadlines:
            st.caption("No sample deadlines in the next 14 days.")
        for deadline in upcoming_deadlines:
            with st.container(border=True):
                st.markdown(f'**{deadline["title"]}**')
                st.caption(f'{deadline["subject"]} · {deadline["due"]} · {deadline["kind"]}')

        st.subheader("Topics to strengthen")
        with st.container(border=True):
            for topic in focus_topics:
                st.markdown(f'- **{topic["name"]}** · {topic["subject"]}')

    with st.expander("AI advisor setup"):
        if groq_api_key_configured():
            st.success("Groq is configured. Open AI Advisor to start a chat.")
        else:
            st.warning(
                "Groq is not configured yet. Add GROQ_API_KEY to your local .env file "
                "to use AI Advisor. No key is needed to explore the rest of the app."
            )


def render_advisor() -> None:
    heading, clear = st.columns([5, 1])
    heading.title("AI Advisor")
    if clear.button("Clear chat", use_container_width=True):
        st.session_state["advisor_messages"] = []
        st.session_state["advisor_message_sources"] = {}
        st.rerun()
    st.write(
        "Ask for explanations, study strategies, or help breaking a topic into smaller steps. "
        "When you add course materials, relevant excerpts are retrieved and shown with the answer."
    )
    st.caption("Advisor requests run through the StudyMate LangGraph workflow.")

    if not groq_api_key_configured():
        st.warning(
            "Add your Groq API key to a local `.env` file as `GROQ_API_KEY=...`, "
            "then restart the app. Your key is never shown in the interface."
        )
        return

    documents, chunks = _material_state()
    profile = get_student_profile()
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["Basic retrieval", "Agentic retrieval + critic"],
        help=(
            "Agentic mode asks Groq to plan alternate searches, then critiques the "
            "grounded answer. It uses extra Groq requests."
        ),
    )
    agentic = retrieval_mode == "Agentic retrieval + critic"
    if agentic:
        st.caption(
            "Agentic mode may use up to two extra Groq requests per question: "
            "one to plan searches and one to critique the answer."
        )
    st.caption(
        f"{len(documents)} material{'s' if len(documents) != 1 else ''} available in this session."
    )
    messages: list[ChatMessage] = st.session_state.setdefault("advisor_messages", [])
    message_sources: dict[int, list[dict]] = st.session_state.setdefault(
        "advisor_message_sources", {}
    )
    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message_sources.get(message_index, [])
            if sources:
                with st.expander(f"Sources used ({len(sources)})"):
                    for source_index, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**[{source_index}] {html.escape(source['source'])}**"
                        )
                        st.caption(source["content"])

    prompt = st.chat_input("What would you like help understanding?")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    updated_history = [*messages, {"role": "user", "content": prompt}]
    workflow_result: dict = {}
    retrieved: list[dict] = []
    search_queries = [prompt]
    reply = ""
    critique_verdict = ""
    with st.chat_message("assistant"):
        with st.spinner("Thinking through it..."):
            try:
                student_context = (
                    f"Program: {profile['program']}\n"
                    f"Term: {profile['term']}\n"
                    f"Learning preferences: {profile['learning_preferences']}\n"
                    f"Advisor memory: {profile['advisor_memory']}"
                )
                workflow_result = run_advisor_workflow(
                    question=prompt,
                    history=updated_history,
                    chunks=chunks,
                    student_context=student_context,
                    agentic=agentic,
                )
                reply = workflow_result["reply"]
                retrieved = workflow_result.get("retrieved", [])
                search_queries = workflow_result.get("search_queries", [prompt])
                critique_verdict = workflow_result.get("critique_verdict", "")
            except GroqAPIError as error:
                messages.append({"role": "user", "content": prompt})
                st.error(str(error))
                if "reply" in locals():
                    st.warning(
                        "A draft response was generated, but the workflow did not finish verification."
                    )
                    st.markdown(reply)
                return
        if documents and not retrieved:
            st.info(
                "No close match was found in your uploaded materials. The advisor will answer "
                "from general knowledge and will not cite a document for this question."
            )
        st.markdown(reply)
        if critique_verdict:
            verdict_messages = {
                "supported": "Source critic: answer supported by the retrieved excerpts.",
                "revised": "Source critic: answer revised to better match the excerpts.",
                "insufficient": "Source critic: the excerpts do not fully answer this question.",
            }
            st.caption(verdict_messages[critique_verdict])
        if retrieved:
            with st.expander(f"Sources used ({len(retrieved)})"):
                for source_index, source in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**[{source_index}] {html.escape(source['source'])}**"
                    )
                    st.caption(source["content"])
    if agentic and chunks:
        with st.expander("Agentic retrieval plan"):
            for query in search_queries:
                st.markdown(f"- {query}")

    assistant_index = len(messages) + 1
    if retrieved:
        message_sources[assistant_index] = retrieved
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
    )


def render_materials() -> None:
    st.title("Materials")
    st.write(
        "Upload course notes to make their text searchable in AI Advisor. "
        "Files are processed locally and kept only for this browser session."
    )
    st.caption(
        "Supported: PDF, TXT, Markdown · Maximum 8 MB per file, 40 MB total, and 20 files · "
        "Scanned PDFs are not OCR'd. Extracted text is sent to Groq only when used as chat context."
    )

    documents, chunks = _material_state()
    uploads = st.file_uploader(
        "Choose study materials",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=_next_material_upload_key(),
        max_upload_size=MAX_FILE_BYTES // (1024 * 1024),
        help=f"Maximum file size: {MAX_FILE_BYTES // (1024 * 1024)} MB each.",
    )

    material_errors: list[str] = []
    if uploads:
        known_ids = {document["id"] for document in documents}
        added_count = 0
        for upload in uploads:
            raw_data = upload.getvalue()
            document_id = hashlib.sha256(raw_data).hexdigest()
            if document_id in known_ids:
                continue
            if len(documents) >= 20:
                material_errors.append("A maximum of 20 materials can be indexed in one session.")
                break
            if sum(document["size"] for document in documents) + len(raw_data) > 40 * 1024 * 1024:
                material_errors.append("The session's total indexed material limit is 40 MB.")
                continue
            try:
                extracted = extract_and_chunk(upload.name, raw_data)
            except MaterialError as error:
                material_errors.append(f"{upload.name}: {error}")
                continue
            documents.append(
                {
                    "id": document_id,
                    "name": upload.name,
                    "size": len(raw_data),
                    "chunk_count": len(extracted),
                }
            )
            chunks.extend(
                {**chunk, "document_id": document_id} for chunk in extracted
            )
            known_ids.add(document_id)
            added_count += 1
        if added_count or material_errors or uploads:
            _reset_material_uploader()
            if added_count:
                st.session_state["materials_flash_success"] = (
                    f"Added {added_count} material{'s' if added_count != 1 else ''} "
                    "to this session's search index."
                )
            if material_errors:
                st.session_state["materials_flash_errors"] = material_errors
            st.rerun()

    flash_success = st.session_state.pop("materials_flash_success", None)
    if flash_success:
        st.success(flash_success)
    flash_errors = st.session_state.pop("materials_flash_errors", [])
    for material_error in flash_errors:
        st.error(material_error)

    if documents:
        st.subheader("Indexed materials")
        for document in list(documents):
            name_col, details_col, remove_col = st.columns([3, 2, 1])
            name_col.markdown(f"**{html.escape(document['name'])}**")
            details_col.caption(
                f"{document['size'] / 1024:.0f} KB · {document['chunk_count']} chunks"
            )
            if remove_col.button("Remove", key=f"remove-{document['id']}"):
                documents[:] = [item for item in documents if item["id"] != document["id"]]
                chunks[:] = [
                    chunk for chunk in chunks if chunk["document_id"] != document["id"]
                ]
                _reset_material_uploader()
                st.rerun()
    else:
        st.info("No materials indexed yet. Uploaded files are not saved after the session ends.")


def render_profile() -> None:
    profile = get_student_profile()
    subjects = get_subject_progress()
    focus_topics = get_focus_topics()
    st.title("My Profile")
    st.write(
        "These details are stored in a local SQLite database on this computer. "
        "Learning preferences and advisor memory are shared with Groq to personalize replies."
    )

    with st.form("student-profile"):
        st.subheader("Student details")
        name = st.text_input("Name", value=profile["first_name"], max_chars=80)
        program = st.text_input("Program", value=profile["program"], max_chars=120)
        term = st.text_input("Current term", value=profile["term"], max_chars=80)
        preferences = st.text_area(
            "How do you prefer to learn?",
            value=profile["learning_preferences"],
            max_chars=2_000,
            help="For example: concise explanations, worked examples, or one question at a time.",
        )
        advisor_memory = st.text_area(
            "What should your advisor remember?",
            value=profile["advisor_memory"],
            max_chars=4_000,
            help="Add context you want used in future replies, such as goals or recurring challenges.",
        )
        if st.form_submit_button("Save profile", type="primary"):
            try:
                save_student_profile(name, program, term, preferences, advisor_memory)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Profile saved locally.")
                st.rerun()

    st.subheader("Subject progress")
    for subject in subjects:
        with st.form(f"subject-{subject['name']}"):
            st.markdown(f"**{subject['name']}**")
            progress_col, hours_col = st.columns(2)
            progress = progress_col.slider(
                "Progress",
                0,
                100,
                value=subject["progress"],
                format="%d%%",
                key=f"progress-{subject['name']}",
            )
            weekly_hours = hours_col.number_input(
                "Study hours this week",
                min_value=0,
                max_value=80,
                value=subject["weekly_hours"],
                key=f"hours-{subject['name']}",
            )
            next_step = st.text_input(
                "Next step",
                value=subject["next_step"],
                max_chars=240,
                key=f"next-{subject['name']}",
            )
            if st.form_submit_button("Save progress"):
                try:
                    save_subject_progress(
                        subject["name"], progress, next_step, weekly_hours
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success(f'{subject["name"]} progress saved.')
                    st.rerun()

    st.subheader("Focus topics")
    for topic in focus_topics:
        name_col, subject_col, action_col = st.columns([3, 2, 1])
        name_col.markdown(f'**{topic["name"]}**')
        subject_col.caption(topic["subject"])
        if action_col.button("Remove", key=f"delete-topic-{topic['id']}"):
            delete_focus_topic(topic["id"])
            st.rerun()

    with st.form("add-focus-topic", clear_on_submit=True):
        topic_col, topic_subject_col, add_col = st.columns([3, 2, 1])
        topic_name = topic_col.text_input("Topic", max_chars=120)
        topic_subject = topic_subject_col.text_input("Subject", max_chars=120)
        add_topic = add_col.form_submit_button("Add topic")
        if add_topic:
            try:
                added = add_focus_topic(topic_name, topic_subject)
            except ValueError as error:
                st.error(str(error))
            else:
                if added:
                    st.success("Focus topic saved.")
                    st.rerun()
                else:
                    st.info("That focus topic is already listed for this subject.")


def render_study_planner() -> None:
    st.title("Study Planner")
    st.write(
        "Create a draft from your saved subject progress and focus topics. "
        "Nothing is added to your saved plan until you approve it."
    )

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    with st.form("study-plan-options"):
        study_days = st.multiselect(
            "Days you can study",
            weekday_names,
            default=weekday_names[:5],
        )
        minutes_per_day = st.slider(
            "Available study time per selected day",
            min_value=25,
            max_value=240,
            value=90,
            step=15,
            format="%d minutes",
        )
        create_draft = st.form_submit_button("Create draft plan", type="primary")

    if create_draft:
        try:
            st.session_state["study_plan_draft"] = build_weekly_draft(
                get_subject_progress(),
                get_focus_topics(),
                [weekday_names.index(day) for day in study_days],
                minutes_per_day,
            )
        except ValueError as error:
            st.error(str(error))

    draft = st.session_state.get("study_plan_draft")
    if draft is not None:
        st.subheader("Draft for your review")
        if not draft:
            st.info("No sessions fit those choices. Select at least one study day.")
        else:
            st.warning(
                "This is only a proposal. Review it, then explicitly approve it to replace "
                "the currently active plan."
            )
            days: dict[str, list[dict]] = {}
            for session in draft:
                days.setdefault(session["study_date"], []).append(session)
            for study_date, sessions in days.items():
                st.markdown(f"**{date.fromisoformat(study_date).strftime('%A, %B %d')}**")
                for session in sessions:
                    st.markdown(
                        f'- {session["duration_minutes"]} min · **{session["subject"]}** — '
                        f'{session["task"]}'
                    )
            approve_col, discard_col = st.columns(2)
            if approve_col.button("Approve and save plan", type="primary", use_container_width=True):
                try:
                    approve_study_plan(draft)
                except ValueError as error:
                    st.error(str(error))
                else:
                    del st.session_state["study_plan_draft"]
                    st.success("Plan approved and saved locally.")
                    st.rerun()
            if discard_col.button("Discard draft", use_container_width=True):
                del st.session_state["study_plan_draft"]
                st.rerun()

    active_plan = get_active_study_plan()
    st.subheader("Approved plan")
    if not active_plan:
        st.info("You do not have an approved study plan yet.")
        return

    completed_count = sum(session["status"] == "completed" for session in active_plan)
    st.progress(completed_count / len(active_plan))
    st.caption(f"{completed_count} of {len(active_plan)} sessions completed")
    for session in active_plan:
        study_date = date.fromisoformat(session["study_date"])
        done = session["status"] == "completed"
        session_col, status_col = st.columns([5, 1])
        label = (
            f'{study_date.strftime("%a, %b %d")} · {session["duration_minutes"]} min · '
            f'**{session["subject"]}** — {session["task"]}'
        )
        session_col.markdown(f'{"~~" if done else ""}{label}{"~~" if done else ""}')
        if status_col.button(
            "Undo" if done else "Done",
            key=f"session-{session['id']}",
        ):
            set_study_session_status(
                session["id"], "planned" if done else "completed"
            )
            st.rerun()


def render_quiz() -> None:
    st.title("Quiz")
    st.write(
        "Generate a practice quiz, answer it, then review your score and explanations. "
        "Answers are evaluated locally and are not sent back to Groq."
    )

    if not groq_api_key_configured():
        st.warning(
            "Add GROQ_API_KEY to your local `.env` file to generate quizzes. "
            "Your key is never displayed in the app."
        )
        return

    with st.form("quiz-generator"):
        topic = st.text_input(
            "Subject or topic",
            value="Data Structures",
            max_chars=300,
            help="For example: graph traversal, eigenvalues, or database normalization.",
        )
        question_count = st.selectbox("Number of questions", [3, 5, 8, 10], index=1)
        generate = st.form_submit_button("Generate quiz", type="primary")

    if generate:
        _, chunks = _material_state()
        try:
            workflow_result = run_quiz_workflow(topic, question_count, chunks)
        except GroqAPIError as error:
            st.error(f"Could not generate the quiz: {error}")
        else:
            questions = workflow_result["questions"]
            st.session_state["quiz_questions"] = questions
            st.session_state["quiz_topic"] = topic.strip()
            st.session_state["quiz_sources"] = workflow_result.get("retrieved", [])
            st.session_state["quiz_id"] = uuid4().hex[:12]
            st.session_state["quiz_submitted"] = False
            st.session_state.pop("quiz_results", None)
            st.rerun()

    questions: list[QuizQuestion] | None = st.session_state.get("quiz_questions")
    if not questions:
        st.info("Choose a topic above to generate your first practice quiz.")
        return

    st.subheader(f'Practice quiz: {st.session_state["quiz_topic"]}')
    sources = st.session_state.get("quiz_sources", [])
    if sources:
        st.caption(f"Questions can draw on {len(sources)} retrieved material excerpts.")
    else:
        st.caption("No matching uploaded excerpts were found; this quiz uses general knowledge.")

    if not st.session_state.get("quiz_submitted", False):
        with st.form("quiz-answers"):
            for question_number, question in enumerate(questions, start=1):
                st.markdown(f"**{question_number}. {question['question']}**")
                st.radio(
                    f"Answer {question_number}",
                    question["options"],
                    index=None,
                    key=f"quiz-{st.session_state['quiz_id']}-{question_number}",
                    label_visibility="collapsed",
                )
            submit_quiz = st.form_submit_button("Submit answers", type="primary")
        if submit_quiz:
            selected_answers = []
            for question_number, question in enumerate(questions, start=1):
                selected = st.session_state.get(
                    f"quiz-{st.session_state['quiz_id']}-{question_number}"
                )
                selected_answers.append(
                    question["options"].index(selected) if selected in question["options"] else None
                )
            results = score_quiz(questions, selected_answers)
            st.session_state["quiz_results"] = results
            st.session_state["quiz_submitted"] = True
            st.rerun()
        return

    results: list[QuizResult] = st.session_state["quiz_results"]
    score = sum(result["is_correct"] for result in results)
    st.metric("Your score", f"{score} / {len(results)}", f"{round(score / len(results) * 100)}%")
    quiz_sources = st.session_state.get("quiz_sources", [])
    for question_number, (question, result) in enumerate(zip(questions, results), start=1):
        st.markdown(
            f"**{question_number}. {question['question']}** — "
            f'{"Correct" if result["is_correct"] else "Review this one"}'
        )
        if result["selected_index"] is None:
            st.caption("No answer selected.")
        elif not result["is_correct"]:
            st.caption(f'Your answer: {question["options"][result["selected_index"]]}')
        st.markdown(f'Correct answer: **{question["options"][result["correct_index"]]}**')
        st.write(question["explanation"])
        if question["source_ids"] and quiz_sources:
            cited_sources = [
                quiz_sources[source_id - 1]
                for source_id in question["source_ids"]
                if source_id <= len(quiz_sources)
            ]
            with st.expander(f"Sources for question {question_number}"):
                for source_id, source in zip(question["source_ids"], cited_sources):
                    st.markdown(f'**[{source_id}] {source["source"]}**')
                    st.caption(source["content"])

    if st.button("Generate another quiz"):
        st.session_state.pop("quiz_questions", None)
        st.session_state.pop("quiz_results", None)
        st.session_state["quiz_submitted"] = False
        st.rerun()


def render_placeholder(page: str) -> None:
    raise ValueError(f"Unknown placeholder page: {page}")


initialize_database()
student_profile = get_student_profile()

with st.sidebar:
    st.markdown("## StudyMate AI")
    st.caption("Your personal study companion")
    page = st.radio(
        "Navigation",
        ["Dashboard", "AI Advisor", "Study Planner", "Quiz", "Materials", "My Profile"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f'{student_profile["program"]} · {student_profile["term"]}')

if page == "Dashboard":
    render_dashboard()
elif page == "AI Advisor":
    render_advisor()
elif page == "Materials":
    render_materials()
elif page == "My Profile":
    render_profile()
elif page == "Study Planner":
    render_study_planner()
elif page == "Quiz":
    render_quiz()
else:
    render_placeholder(page)
