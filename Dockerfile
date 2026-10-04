FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system studymate \
    && useradd --system --gid studymate --home-dir /app studymate \
    && mkdir -p /app/data \
    && chown -R studymate:studymate /app

COPY --chown=studymate:studymate app.py README.md ./
COPY --chown=studymate:studymate studymate ./studymate
COPY --chown=studymate:studymate .streamlit ./.streamlit

USER studymate
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py"]
