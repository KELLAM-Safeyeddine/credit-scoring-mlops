FROM apache/airflow:2.11.0-python3.11
USER root
RUN python -m venv /opt/creditS && chown -R airflow:root /opt/creditS
USER airflow
COPY airflow/requirements-worker*.txt /tmp/
RUN --mount=type=bind,source=.docker-wheels,target=/tmp/credit-wheels \
    if [ -f /tmp/credit-wheels/.complete ]; then CREDIT_PIP_INDEX=--no-index; else CREDIT_PIP_INDEX=''; fi \
    && /opt/creditS/bin/python -m pip install $CREDIT_PIP_INDEX --no-cache-dir --find-links /tmp/credit-wheels \
    -r /tmp/requirements-worker.txt -c /tmp/requirements-worker-lock.txt \
    && /opt/creditS/bin/python -m pip check
COPY airflow/requirements.txt /tmp/requirements-airflow.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.11.0/constraints-3.11.txt
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow
ENV CREDIT_WORKER_PYTHON=/opt/creditS/bin/python
