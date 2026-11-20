import json
import time
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-classifier-prod"
STUDENT_USN = "1MS25CI058"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    with open("outputs/metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    winner_run_id = summary["winner_run_id"]
    loser_run_id = summary["loser_run_id"]
    winner_hf_id = summary["winner_hf_id"]
    loser_hf_id = summary["loser_hf_id"]

    print(f"Registering Champion run ({winner_run_id}) -> {winner_hf_id}")
    champ_mv = mlflow.register_model(
        model_uri=f"runs:/{winner_run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )

    print(f"Registering Challenger run ({loser_run_id}) -> {loser_hf_id}")
    chall_mv = mlflow.register_model(
        model_uri=f"runs:/{loser_run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )

    time.sleep(2)

    for mv, hf_id, role in [
        (champ_mv, winner_hf_id, "champion"),
        (chall_mv, loser_hf_id, "challenger"),
    ]:
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            key="hf_model_id",
            value=hf_id,
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            key="role",
            value=role,
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            key="student_usn",
            value=STUDENT_USN,
        )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=champ_mv.version,
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=chall_mv.version,
    )

    print("\n==========================================")
    print(f"Model Registry: {REGISTERED_MODEL_NAME}")
    print(f"  @champion   -> Version {champ_mv.version} ({winner_hf_id})")
    print(f"  @challenger -> Version {chall_mv.version} ({loser_hf_id})")
    print("==========================================")


if __name__ == "__main__":
    main()
