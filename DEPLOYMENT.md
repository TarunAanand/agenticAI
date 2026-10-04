# Deployment

## Docker Compose

Requires Docker Engine and Docker Compose v2.

1. Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Keep this file private; it is excluded from the Docker build context.
2. Build and start StudyMate:

   ```powershell
   docker compose up --build -d
   ```

3. Open `http://localhost:8501`. Check startup with `docker compose ps` and logs with `docker compose logs -f studymate`.
4. Stop the app with `docker compose down`. The named `studymate-data` volume preserves the local SQLite profile and approved plans. `docker compose down -v` deletes that data volume.

The container runs as a non-root user and has a Streamlit health check. Compose binds to `127.0.0.1` by default so it is not exposed to the local network. Set `GROQ_MODEL` in `.env` to override the default. For a hosted container service, provide `GROQ_API_KEY` and `GROQ_MODEL` through the platform's secret/environment configuration, expose port `8501` only behind an authenticated access layer, and attach persistent storage at `/app/data`. Do not bake secrets into the image.

## Hosting considerations

- The included app has no authentication or multi-user isolation. Do not expose it publicly with real student data until access control and per-user storage are implemented.
- Local Compose binds to `127.0.0.1` by default. Hosted deployments must add an authenticated access layer and per-user data isolation before making the service reachable to other users.
- The SQLite database is local and unencrypted. Use private storage and restrict access to the deployment volume.
- Uploaded materials and chat/quiz attempts are held in the browser session; they are not persisted or shared between sessions.
- Extracted material excerpts and saved advisor preferences/memory are sent to Groq when used. Review Groq's current data-retention terms before using sensitive information.
- Provider evaluation runs make real calls only when `scripts/evaluate.py --live` is supplied; see the README before running one.

## Manual Streamlit deployment

For Streamlit Community Cloud, select this repository and set `app.py` as the entry point. Configure `GROQ_API_KEY` in the app's secret/environment settings rather than committing `.env`. Community Cloud filesystems may be ephemeral, so local SQLite changes should not be assumed durable; use a persistent host or managed database before relying on saved profiles or plans.
