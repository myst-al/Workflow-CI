"""Training entry point for the MLflow Project used by the CI workflow.

Logs to the local ./mlruns tracking store (default MLflow behaviour when no
tracking URI is set) so the CI runner can look up the resulting run_id and
build a Docker image from it.
"""
import argparse
import os
import sys

# MLflow prints emoji (e.g. the run-URL banner) to stdout; Windows consoles
# often default to cp1252, which can't encode them and crashes the run.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMN = "Churn"
DATA_DIR = os.path.join(os.path.dirname(__file__), "telco_customer_churn_preprocessing")


def load_split(name: str):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    # Named max_depth_arg (not max_depth) so MLflow Project's own logging of
    # this entry-point parameter doesn't collide with autolog separately
    # logging the fitted model's actual `max_depth` hyperparameter.
    parser.add_argument("--max_depth_arg", type=int, default=-1)
    args = parser.parse_args()
    max_depth = None if args.max_depth_arg == -1 else args.max_depth_arg

    # Experiment is chosen by `mlflow run --experiment-name` (see ci.yml), not
    # here - calling set_experiment() would conflict with the run ID that
    # `mlflow run` already injected via the MLFLOW_RUN_ID env var.
    mlflow.sklearn.autolog()

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"precision: {precision_score(y_test, y_pred):.4f}")
        print(f"recall: {recall_score(y_test, y_pred):.4f}")
        print(f"f1_score: {f1_score(y_test, y_pred):.4f}")
        print(f"RUN_ID:{run.info.run_id}")


if __name__ == "__main__":
    main()
