FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib/Pillow (rendering GIFs/plots headlessly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

COPY run.py ./
COPY tests ./tests

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

CMD ["streamlit", "run", "src/trebuchet_sim/web/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
