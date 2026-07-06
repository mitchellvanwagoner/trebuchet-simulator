# Stage 1: build the wheel (keeps setuptools and the src tree out of the runtime image)
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --no-cache-dir -w /wheels .

# Stage 2: runtime
#
# 3.12, not 3.11: docker-constraints.txt pins the exact versions from the dev
# venv, and scipy's pinned version requires Python >=3.12 (it publishes no
# cp311 wheels at all) - keep this in step with whatever scipy pin the
# constraints file carries.
FROM python:3.12-slim

# numba installed alongside the wheel so the optimizer uses the JIT-vectorized
# engine instead of the slow per-simulation multiprocessing fallback. Versions
# are pinned by the constraints file so published images are reproducible.
COPY docker-constraints.txt /tmp/docker-constraints.txt
COPY --from=builder /wheels /tmp/wheels
RUN pip install --no-cache-dir -c /tmp/docker-constraints.txt "$(ls /tmp/wheels/*.whl)[fast]" \
    && rm -rf /tmp/wheels /tmp/docker-constraints.txt

# Non-root runtime user. /app/data holds the dashboard's saved defaults
# (user_defaults.json); mount a volume there to persist them across recreations.
RUN useradd --create-home --uid 1000 trebuchet \
    && mkdir -p /app/data \
    && chown trebuchet:trebuchet /app/data
COPY docker-entrypoint.py /usr/local/bin/docker-entrypoint.py
WORKDIR /app

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TREBUCHET_DATA_DIR=/app/data \
    MPLBACKEND=Agg

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Container starts as root so the entrypoint can chown /app/data regardless of
# what a bind mount brought with it, then drops to the trebuchet user before
# running the actual command - see docker-entrypoint.py.
ENTRYPOINT ["python", "/usr/local/bin/docker-entrypoint.py"]
CMD ["trebuchet-web"]
