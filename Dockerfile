FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-semantic.txt ./

ARG INSTALL_SEMANTIC=false
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_SEMANTIC" = "true" ]; then \
        python -m pip install --no-cache-dir -r requirements-semantic.txt; \
    fi

COPY app.py build_artifacts.py evaluate_recommendations.py ./
COPY src ./src

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
