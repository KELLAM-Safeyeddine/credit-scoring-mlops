# syntax=docker/dockerfile:1
FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS dependencies
RUN python -m venv /opt/creditS
COPY api/requirements*.txt /tmp/
RUN --mount=type=bind,source=.docker-wheels,target=/tmp/credit-wheels \
    if [ -f /tmp/credit-wheels/.complete ]; then CREDIT_PIP_INDEX=--no-index; else CREDIT_PIP_INDEX=''; fi \
    && /opt/creditS/bin/python -m pip install $CREDIT_PIP_INDEX --no-cache-dir \
       --find-links /tmp/credit-wheels -r /tmp/requirements.txt -c /tmp/requirements-lock.txt \
    && /opt/creditS/bin/python -m pip check

FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 credit && useradd --uid 10001 --gid credit --create-home credit
COPY --from=dependencies /opt/creditS /opt/creditS
WORKDIR /app
COPY --chown=credit:credit api /app/api
COPY --chown=credit:credit src /app/src
COPY --chown=credit:credit params.yaml /app/params.yaml
RUN mkdir -p /app/runtime/api && chown -R credit:credit /app/runtime
ENV PATH=/opt/creditS/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    MLFLOW_TRACKING_URI=http://mlflow:5000
USER credit
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
    CMD ["/opt/creditS/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=3)"]
CMD ["/opt/creditS/bin/python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
