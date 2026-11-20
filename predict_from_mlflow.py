import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

# NOTE: No Hugging Face imports or pipelines exist in this client script.
# All inference runs strictly through the MLflow Model Registry alias URI.

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-classifier-prod"
ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=ALIAS)
    run = client.get_run(mv.run_id)
    hf_model_id = mv.tags.get("hf_model_id") or run.data.params.get("hf_model_id", "unknown")

    print("==========================================")
    print(f"Loaded URI        : {MODEL_URI}")
    print(f"Registry Version  : v{mv.version}")
    print(f"Source Run ID     : {mv.run_id}")
    print(f"Hugging Face ID   : {hf_model_id}")
    print("==========================================\n")

    model = mlflow.pyfunc.load_model(MODEL_URI)

    new_df = pd.read_csv("data/new_data.csv")
    preds_df = model.predict(new_df[["text"]])

    preds_df.insert(0, "registry_alias", f"@{ALIAS}")
    preds_df.insert(1, "registry_version", f"v{mv.version}")

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/new_data_predictions.csv"
    preds_df.to_csv(output_path, index=False)

    print(preds_df.to_string(index=False))
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
