# StudyMate AI

A small, runnable Streamlit app for a student academic advisor. The AI Advisor chats with Groq and can retrieve from uploaded course materials; student profile, learning preferences, focus topics, and subject progress are stored locally in SQLite.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

The app runs without an API key, but the AI Advisor requires one. Put your own key in `.env` as `GROQ_API_KEY=...`; optionally set `GROQ_MODEL` to use a different Groq chat model. Keep `.env` private; it is ignored by git. Never put a real key in source code or `.env.example`.

## Template architecture

- `app.py` contains the Streamlit shell, navigation, dashboard, and page previews.
- `studymate/data.py` provides replaceable sample student, subject, deadline, and focus-topic data.
- `studymate/config.py` loads local `.env` configuration and exposes a safe key-presence check.
- `studymate/groq_client.py` isolates the Groq chat-completions request, bounded conversation context, optional retrieved excerpts, and provider errors. The API key is sent only to Groq and is never displayed.
- `studymate/materials.py` extracts PDF/TXT/Markdown and retrieves relevant text with a small in-memory TF-IDF cosine-similarity index. No embedding model or external vector database is used.
- `studymate/workflows.py` uses LangGraph state graphs to orchestrate advisor query planning, local retrieval, answer generation, optional source criticism, and quiz retrieval/generation.
- `evaluations/cases.json` and `scripts/evaluate.py` provide a synthetic, dry-run-by-default comparison of no/basic/agentic RAG modes across up to three models.
- Uploaded files and the index live only in Streamlit session state. Limits are 8 MB per file, 40 MB per session, 20 materials, and 2,000 chunks per document. Retrieval drops weak lexical matches (cosine score below 0.12). Extracted snippets are sent to Groq only as context for an advisor question; deleting a material removes it from future retrieval. Scanned PDFs require OCR, which is not included.
- `studymate/storage.py` stores the profile, advisor memory/preferences, subject progress, focus topics, and explicitly approved study plans in `data/studymate.sqlite3` on this computer. The database is ignored by git and is not encrypted; protect the local machine and back up this file if needed. Advisor memory/preferences are sent to Groq for personalization when you chat.
- Advisor conversations, quiz attempts, and uploaded material remain session-only and are cleared when the Streamlit session ends.
- In **Agentic retrieval + critic** mode, Groq proposes up to three lexical search queries, the app searches local chunks, then Groq critiques the draft against retrieved excerpts. This adds up to two provider requests when materials are indexed. Use Basic retrieval to avoid these extra requests.
- Run the lightweight tests with `python -m unittest discover -s tests -v`.

## Retrieval/model evaluation

The offline fixture at `evaluations/cases.json` contains synthetic study-note passages, questions, expected phrases, and expected source files. Preview the run without network access or provider cost:

```powershell
python scripts/evaluate.py
```

For a live comparison, specify 2-3 models available to your Groq account. This makes real API calls: with the default four cases, expect up to 20 calls per model (up to 60 for three models), including query planning and answer critique for agentic RAG.

```powershell
$env:GROQ_EVAL_MODELS = "model-one,model-two"
python scripts/evaluate.py --live --output eval-results\comparison.json
```

The report compares no RAG, basic local TF-IDF RAG, and agentic query-planning/critic RAG for each model. It records latency, retrieval hit rate, citation rate, and expected-phrase recall. Phrase recall is a transparent lexical proxy, not semantic grading; inspect the per-case answers and use a human review before drawing quality conclusions. Quiz answers and any real student profile data are not part of this synthetic evaluation.

## Incremental roadmap

1. App template and dashboard: navigation, mock progress, subjects, exams/deadlines, and weak topics.
2. Groq-powered advisor chat: safe configuration, multi-turn session chat, and clear missing-key/error states.
3. Materials uploads with basic local TF-IDF RAG: PDF/TXT/Markdown ingestion, session-only index, and source references.
4. Student profile, memory, progress, and local SQLite persistence.
5. Agentic retrieval with query planning and an answer critic against retrieved excerpts.
6. Study planner with human approval before applying changes: local weekly drafts, explicit approval before saving, and per-session completion tracking.
7. Quiz generation and evaluation: Groq-generated multiple-choice quizzes with local scoring and answer explanations.
8. LangGraph orchestration across advisor and quiz workflows.
9. Evaluation across no/basic/agentic RAG and 2-3 models: dry-run by default, live API calls require `--live`.
10. Product polish and deployment (current): escaped profile-name rendering, Streamlit production settings, non-root Docker image, health check, and persistent Compose volume.

Each increment should stay independently runnable and add only the dependencies it needs. In Study Planner, the preview is local and does not call Groq; only clicking **Approve and save plan** stores it as the active plan. Approving replaces the active plan while leaving prior plan records in the local database. Quiz generation uses Groq and may include retrieved excerpts; answers are scored locally and are not sent to the model.

## Deployment and security notes

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose and hosting instructions. The included container runs as non-root, has a health check, and stores SQLite data in a named volume. This app currently has no sign-in or per-user isolation; do not expose it publicly with real student data until those are added.
