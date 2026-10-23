FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN grep -v '^sentence-transformers' requirements.txt > /tmp/requirements-base.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-base.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir 'sentence-transformers>=5.0'

ENV HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app.py build_artifacts.py evaluate_recommendations.py ./
COPY src ./src
COPY job_recommender.joblib ./
ENV JOB_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
