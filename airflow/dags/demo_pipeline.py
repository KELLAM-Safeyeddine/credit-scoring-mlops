"""Synthetic lifecycle DAGs with a real monitoring-triggered retraining run."""

from datetime import UTC, datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from demo_worker import demo_worker

OPTIONS = dict(
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=20)},
    tags=["credit", "synthetic-demo"],
)


def workspace() -> str:
    """Read only a unique workspace name; private credentials stay in its file."""
    conf = get_current_context()["dag_run"].conf
    uid = conf.get("demo_id", "")
    if set(conf) != {"demo_id"} or len(uid) != 12 or any(c not in "0123456789abcdef" for c in uid):
        raise ValueError("Invalid demonstration identifier.")
    return "/workspace/runtime/demo/" + uid


@dag(dag_id="credit_demo_retraining", **OPTIONS)
def retraining():
    @task(pool="credit_compute")
    def train_v2():
        result = demo_worker("train_v2", workspace())
        assert result["gate_status"] == "rejected"
        return result

    @task(pool="credit_compute")
    def train_v3():
        result = demo_worker("train_v3", workspace())
        assert result["gate_status"] == "accepted"
        return result

    @task
    def reload_api():
        return demo_worker("deploy_v3", workspace())

    train_v2() >> train_v3() >> reload_api()


@dag(dag_id="credit_demo", **OPTIONS)
def demonstration():
    @task(pool="credit_compute")
    def version_data():
        return demo_worker("validate_fixture", workspace())

    @task(pool="credit_compute")
    def train_v1():
        return demo_worker("train_v1", workspace())

    @task
    def deploy_v1():
        return demo_worker("deploy_v1", workspace())

    @task.branch(pool="credit_compute")
    def monitor_batch():
        result = demo_worker("monitor", workspace())
        return "trigger_retraining" if result["trigger_retraining"] else "no_alert"

    @task
    def no_alert():
        raise RuntimeError("The controlled scenario must produce a drift alert.")

    trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="credit_demo_retraining",
        trigger_run_id="retraining__{{ run_id }}",
        conf={"demo_id": "{{ dag_run.conf.demo_id }}"},
        wait_for_completion=True,
        poke_interval=5,
        pool="default_pool",
    )
    version_data() >> train_v1() >> deploy_v1() >> monitor_batch() >> [trigger, no_alert()]


retraining()
demonstration()
