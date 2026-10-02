"""Train, audit, explain and branch before any Registry promotion."""

from datetime import UTC, datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator, get_current_context
from airflow.utils.trigger_rule import TriggerRule
from worker_client import run_worker


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=20),
        "execution_timeout": timedelta(minutes=90),
        "pool": "credit_compute",
    },
    tags=["credit", "training"],
)
def training_pipeline():
    @task
    def ingest() -> str:
        context = get_current_context()
        return run_worker(
            "ingest", run_id="training:" + context["run_id"], conf=context["dag_run"].conf
        )["workspace"]

    @task
    def validate_data(workspace: str) -> None:
        run_worker("validate_data", workspace)

    @task
    def build_features(workspace: str) -> None:
        run_worker("build_features", workspace)

    @task
    def train_models(workspace: str) -> None:
        run_worker("train_models", workspace)

    @task
    def evaluate_performance(workspace: str) -> dict:
        return run_worker("evaluate_performance", workspace)

    @task
    def evaluate_fairness(workspace: str) -> None:
        run_worker("evaluate_fairness", workspace)

    @task
    def compute_shap(workspace: str) -> None:
        run_worker("compute_shap", workspace)

    def branch(workspace: str) -> str:
        return (
            "register_and_promote"
            if run_worker("quality_gate", workspace)["passed"]
            else "reject_model"
        )

    @task
    def register_and_promote(workspace: str) -> dict:
        return run_worker("register_and_promote", workspace)

    @task
    def reject_model(workspace: str) -> dict:
        return run_worker("reject_model", workspace)

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def notification(workspace: str) -> dict:
        return run_worker("notification", workspace)

    workspace = ingest()
    validation = validate_data(workspace)
    features = build_features(workspace)
    training = train_models(workspace)
    performance = evaluate_performance(workspace)
    fairness = evaluate_fairness(workspace)
    explanations = compute_shap(workspace)
    gate = BranchPythonOperator(task_id="quality_gate", python_callable=branch, op_args=[workspace])
    validation >> features >> training >> performance >> fairness >> explanations >> gate
    gate >> [register_and_promote(workspace), reject_model(workspace)] >> notification(workspace)


training_pipeline()
