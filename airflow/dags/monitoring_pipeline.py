"""Evaluate production alerts and trigger the gated training DAG."""

from datetime import UTC, datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from worker_client import run_worker


@dag(
    dag_id="monitoring_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=20),
        "execution_timeout": timedelta(minutes=30),
        "pool": "credit_compute",
    },
    tags=["credit", "monitoring"],
)
def monitoring_pipeline():
    @task
    def load_production_batch() -> str:
        context = get_current_context()
        return run_worker(
            "load_production_batch",
            run_id="monitoring:" + context["run_id"],
            conf=context["dag_run"].conf,
        )["workspace"]

    @task
    def drift_report(workspace: str) -> None:
        run_worker("drift_report", workspace)

    @task
    def fairness_monitor(workspace: str) -> None:
        run_worker("fairness_monitor", workspace)

    @task
    def evaluate_alert_rules(workspace: str) -> dict:
        return run_worker("evaluate_alert_rules", workspace)

    def choose(result: dict) -> str:
        return "trigger_training" if result["trigger_retraining"] else "no_alert"

    workspace = load_production_batch()
    drift = drift_report(workspace)
    fairness = fairness_monitor(workspace)
    alerts = evaluate_alert_rules(workspace)
    workspace >> drift >> fairness >> alerts
    branch = BranchPythonOperator(task_id="alert_branch", python_callable=choose, op_args=[alerts])
    trigger = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        trigger_run_id="monitoring__{{ run_id }}",
        conf={"source_monitoring_run": "{{ run_id }}"},
        wait_for_completion=False,
        skip_when_already_exists=True,
    )
    alerts >> branch >> [trigger, EmptyOperator(task_id="no_alert")]


monitoring_pipeline()
