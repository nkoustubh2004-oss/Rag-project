import mlflow
import time

# -------- INIT --------
mlflow.set_tracking_uri("file:./mlruns")


# -------- RUN CONTROL --------
def start_run(run_name="Run"):
    mlflow.set_experiment("RAG_Experiment")

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.start_run(run_name=run_name)


def end_run():
    if mlflow.active_run():
        mlflow.end_run()


# -------- LOGGING --------
def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def log_text(text, filename="output.txt"):
    mlflow.log_text(text, filename)


# -------- TIMER --------
def start_timer():
    return time.time()


def end_timer(start):
    return time.time() - start