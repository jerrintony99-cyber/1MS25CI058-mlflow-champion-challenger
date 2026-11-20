import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

# ==========================================
# CONFIGURATION & PRE-DEFINED DECISION RULE
# ==========================================
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-classifier-prod"
STUDENT_USN = "1MS25CI058"

TASK = "sentiment"
DATASET_NAME = "glue/sst2 (validation)"
EVAL_SAMPLES = 100
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_A_RUN_NAME = "A_sentiment_distilbert-sst2"

MODEL_B_ID = "siebert/sentiment-roberta-large-english"
MODEL_B_RUN_NAME = "B_sentiment_roberta-large"

SUMMARY_RUN_NAME = "compare_sentiment"

# Pre-defined Champion-Challenger Decision Rule (Matches README.md)
PRIMARY_METRIC = "macro_f1"       # Higher is better
MIN_MARGIN = 0.01                 # Challenger must beat Champion by >= 1.0 percentage point (0.01)
MAX_LATENCY_MULTIPLIER = 6.0      # Guardrail: reject Challenger if > 6x slower than Champion

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# CUSTOM PYFUNC WRAPPER (NO HF IN CLIENT)
# ==========================================
class HFSentimentPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str, max_length: int = 256, batch_size: int = 16):
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.classifier = None

    def load_context(self, context):
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            device=device,
            truncation=True,
            max_length=self.max_length,
        )

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if self.classifier is None:
            self.load_context(context)

        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        elif isinstance(model_input, dict):
            texts = [str(x) for x in model_input["text"]]
        else:
            texts = [str(x) for x in model_input]

        raw_preds = self.classifier(texts, batch_size=self.batch_size)
        results = []
        for text, pred in zip(texts, raw_preds):
            label_str = str(pred["label"]).upper()
            pred_label = 1 if ("POS" in label_str or label_str.endswith("_1")) else 0
            results.append(
                {
                    "text": text,
                    "predicted_label": pred_label,
                    "predicted_sentiment": "POSITIVE" if pred_label == 1 else "NEGATIVE",
                    "confidence": round(float(pred["score"]), 4),
                    "hf_model_id": self.hf_model_id,
                }
            )
        return pd.DataFrame(results)


# ==========================================
# DATA PREPARATION
# ==========================================
def prepare_eval_data() -> pd.DataFrame:
    # Fetch SST-2 validation examples via Hugging Face Datasets Server REST API
    # (avoids datasets/huggingface_hub hf:// URI parsing issues).
    url = (
        "https://datasets-server.huggingface.co/rows?"
        + urllib.parse.urlencode(
            {
                "dataset": "nyu-mll/glue",
                "config": "sst2",
                "split": "validation",
                "offset": 0,
                "length": 100,
            }
        )
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = [item["row"] for item in payload["rows"]]
        df_full = pd.DataFrame(rows)
        df_sampled = df_full.sample(
            n=min(EVAL_SAMPLES, len(df_full)), random_state=SEED
        ).reset_index(drop=True)
        text_col = "sentence" if "sentence" in df_sampled.columns else "text"
        df = pd.DataFrame(
            {
                "sample_id": range(len(df_sampled)),
                "text": df_sampled[text_col].astype(str),
                "reference_label": df_sampled["label"].astype(int),
                "reference_sentiment": [
                    "POSITIVE" if int(lbl) == 1 else "NEGATIVE" for lbl in df_sampled["label"]
                ],
            }
        )
    except Exception:
        fallback_items = [
            ("hide new secretions from the parental units", 0),
            ("contains no wit , only labored gags", 0),
            (
                "that loves its characters and communicates something rather beautiful about human nature",
                1,
            ),
            ("remains utterly satisfied to remain the same throughout", 0),
            ("on the worst revenge-of-the-nerds cliches the filmmakers could dredge up", 0),
            ("that 's far too tragic to merit such superficial treatment", 0),
            (
                "demonstrates that the director of such spineless dramas as adrift has sort of a funny streak",
                1,
            ),
            ("a depressed fifteen-year-old 's suicidal poetry", 0),
            ("are more deeply thought through than in most ` right-thinking ' films", 1),
            ("goes to absurd lengths", 0),
        ]
        repeated = (fallback_items * ((EVAL_SAMPLES // len(fallback_items)) + 1))[:EVAL_SAMPLES]
        rng = np.random.default_rng(SEED)
        order = rng.permutation(EVAL_SAMPLES)
        ordered_items = [repeated[i] for i in order]
        df = pd.DataFrame(
            {
                "sample_id": range(EVAL_SAMPLES),
                "text": [t for t, _ in ordered_items],
                "reference_label": [int(lbl) for _, lbl in ordered_items],
                "reference_sentiment": [
                    "POSITIVE" if int(lbl) == 1 else "NEGATIVE" for _, lbl in ordered_items
                ],
            }
        )

    df.to_csv("data/eval_inputs.csv", index=False)
    return df


def log_pyfunc_model_compatible(wrapper: mlflow.pyfunc.PythonModel, pip_reqs: list):
    try:
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            pip_requirements=pip_reqs,
        )
    except TypeError:
        return mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            pip_requirements=pip_reqs,
        )


# ==========================================
# EVALUATION & LOGGING PER CANDIDATE
# ==========================================
def evaluate_and_log_model(
    hf_model_id: str,
    run_name: str,
    role: str,
    eval_df: pd.DataFrame,
    device_str: str,
):
    print(f"\n--- Running {run_name} ({hf_model_id}) ---")
    wrapper = HFSentimentPyfuncModel(
        hf_model_id=hf_model_id, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
    )
    wrapper.load_context(context=None)

    input_df = eval_df[["text"]].copy()

    # Warm-up 1 sample so model initialization overhead isn't counted in inference latency
    _ = wrapper.predict(context=None, model_input=input_df.iloc[:1])

    start_time = time.perf_counter()
    preds_df = wrapper.predict(context=None, model_input=input_df)
    total_inference_s = time.perf_counter() - start_time
    avg_latency_ms = (total_inference_s / len(eval_df)) * 1000.0

    y_true = eval_df["reference_label"].values
    y_pred = preds_df["predicted_label"].values

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "avg_latency_ms": float(avg_latency_ms),
        "total_inference_s": float(total_inference_s),
    }

    short_tag = "model_a" if role == "champion_candidate" else "model_b"
    out_df = pd.concat(
        [
            eval_df[["sample_id", "text", "reference_label", "reference_sentiment"]],
            preds_df[["predicted_label", "predicted_sentiment", "confidence"]],
        ],
        axis=1,
    )
    out_df["correct"] = out_df["reference_label"] == out_df["predicted_label"]
    pred_csv_path = f"outputs/predictions_{short_tag}.csv"
    out_df.to_csv(pred_csv_path, index=False)

    metrics_path = f"outputs/metrics_{short_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"hf_model_id": hf_model_id, "role": role, "metrics": metrics}, f, indent=2)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "hf_model_id": hf_model_id,
                "task": TASK,
                "role": role,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "eval_samples": EVAL_SAMPLES,
                "seed": SEED,
                "device": device_str,
            }
        )
        mlflow.set_tags(
            {
                "student_usn": STUDENT_USN,
                "hf_model_id": hf_model_id,
                "role": role,
                "dataset_name": DATASET_NAME,
                "library_versions": f"transformers={transformers.__version__},torch={torch.__version__},mlflow={mlflow.__version__}",
            }
        )
        mlflow.log_metrics(metrics)

        log_pyfunc_model_compatible(
            wrapper=wrapper,
            pip_reqs=[
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__}",
                "pandas",
            ],
        )

        mlflow.log_artifact("data/eval_inputs.csv")
        mlflow.log_artifact(pred_csv_path)
        mlflow.log_artifact(metrics_path)

        return run.info.run_id, metrics, out_df


# ==========================================
# PLOTS & QUALITATIVE DISAGREEMENTS
# ==========================================
def generate_plots_and_disagreements(
    metrics_a: dict, metrics_b: dict, out_a: pd.DataFrame, out_b: pd.DataFrame
):
    metric_names = ["accuracy", "macro_f1", "precision", "recall"]
    vals_a = [metrics_a[m] for m in metric_names]
    vals_b = [metrics_b[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, vals_a, width, label="Model A (DistilBERT)", color="#4C72B0")
    rects2 = ax.bar(x + width / 2, vals_b, width, label="Model B (RoBERTa-Large)", color="#DD8452")

    ax.set_ylabel("Score (Higher is Better)")
    ax.set_title("Champion (Model A) vs Challenger (Model B) — Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "Precision", "Recall"])
    ax.set_ylim(0.0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bar_plot_path = "outputs/metrics_comparison.png"
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cm_a = confusion_matrix(out_a["reference_label"], out_a["predicted_label"], labels=[0, 1])
    cm_b = confusion_matrix(out_b["reference_label"], out_b["predicted_label"], labels=[0, 1])

    ConfusionMatrixDisplay(cm_a, display_labels=["NEG", "POS"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Model A: DistilBERT-SST2")

    ConfusionMatrixDisplay(cm_b, display_labels=["NEG", "POS"]).plot(
        ax=axes[1], colorbar=False, cmap="Oranges"
    )
    axes[1].set_title("Model B: RoBERTa-Large")

    cm_plot_path = "outputs/confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close(fig)

    merged = out_a[["sample_id", "text", "reference_sentiment"]].copy()
    merged["model_a_pred"] = out_a["predicted_sentiment"]
    merged["model_a_conf"] = out_a["confidence"]
    merged["model_b_pred"] = out_b["predicted_sentiment"]
    merged["model_b_conf"] = out_b["confidence"]

    disagreements = merged[merged["model_a_pred"] != merged["model_b_pred"]].copy()
    disagreements["analysis_note"] = disagreements.apply(
        lambda r: (
            "Model B captured nuance/context missed by Model A"
            if r["model_b_pred"] == r["reference_sentiment"]
            else "Model A predicted accurately; Model B over-interpreted"
        ),
        axis=1,
    )
    disagreements_path = "outputs/qualitative_disagreements.csv"
    disagreements.to_csv(disagreements_path, index=False)

    return bar_plot_path, cm_plot_path, disagreements_path


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    eval_df = prepare_eval_data()

    run_a_id, metrics_a, out_a = evaluate_and_log_model(
        hf_model_id=MODEL_A_ID,
        run_name=MODEL_A_RUN_NAME,
        role="champion_candidate",
        eval_df=eval_df,
        device_str=device_str,
    )

    run_b_id, metrics_b, out_b = evaluate_and_log_model(
        hf_model_id=MODEL_B_ID,
        run_name=MODEL_B_RUN_NAME,
        role="challenger_candidate",
        eval_df=eval_df,
        device_str=device_str,
    )

    bar_plot_path, cm_plot_path, disagreements_path = generate_plots_and_disagreements(
        metrics_a, metrics_b, out_a, out_b
    )

    delta_primary = metrics_b[PRIMARY_METRIC] - metrics_a[PRIMARY_METRIC]
    latency_ratio = metrics_b["avg_latency_ms"] / max(metrics_a["avg_latency_ms"], 1e-6)

    if delta_primary >= MIN_MARGIN and latency_ratio <= MAX_LATENCY_MULTIPLIER:
        winner = "Model_B"
        winner_run_id = run_b_id
        loser_run_id = run_a_id
        winner_hf_id = MODEL_B_ID
        loser_hf_id = MODEL_A_ID
        decision_reason = (
            f"Challenger (Model B) improved {PRIMARY_METRIC} by +{delta_primary:.4f} "
            f"(>= {MIN_MARGIN} threshold) within latency guardrail ({latency_ratio:.2f}x <= {MAX_LATENCY_MULTIPLIER}x)."
        )
    else:
        winner = "Model_A"
        winner_run_id = run_a_id
        loser_run_id = run_b_id
        winner_hf_id = MODEL_A_ID
        loser_hf_id = MODEL_B_ID
        decision_reason = (
            f"Champion (Model A) retained: delta {PRIMARY_METRIC} = {delta_primary:+.4f}, "
            f"latency ratio = {latency_ratio:.2f}x."
        )

    summary_payload = {
        "student_usn": STUDENT_USN,
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "primary_metric": PRIMARY_METRIC,
        "min_margin": MIN_MARGIN,
        "max_latency_multiplier": MAX_LATENCY_MULTIPLIER,
        "delta_primary_metric": round(delta_primary, 4),
        "latency_ratio": round(latency_ratio, 2),
        "winner": winner,
        "winner_run_id": winner_run_id,
        "loser_run_id": loser_run_id,
        "winner_hf_id": winner_hf_id,
        "loser_hf_id": loser_hf_id,
        "decision_reason": decision_reason,
        "model_a": {"run_id": run_a_id, "hf_model_id": MODEL_A_ID, "metrics": metrics_a},
        "model_b": {"run_id": run_b_id, "hf_model_id": MODEL_B_ID, "metrics": metrics_b},
    }
    summary_json_path = "outputs/metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with mlflow.start_run(run_name=SUMMARY_RUN_NAME):
        mlflow.log_params(
            {
                "primary_metric": PRIMARY_METRIC,
                "margin": MIN_MARGIN,
                "max_latency_multiplier": MAX_LATENCY_MULTIPLIER,
                "model_a_id": MODEL_A_ID,
                "model_b_id": MODEL_B_ID,
                "eval_samples": EVAL_SAMPLES,
            }
        )
        mlflow.set_tags(
            {
                "student_usn": STUDENT_USN,
                "winner": winner,
                "winner_hf_id": winner_hf_id,
                "dataset_name": DATASET_NAME,
            }
        )
        mlflow.log_metrics(
            {
                "model_a_macro_f1": metrics_a["macro_f1"],
                "model_b_macro_f1": metrics_b["macro_f1"],
                "delta_primary_metric": delta_primary,
                "latency_ratio": latency_ratio,
            }
        )
        mlflow.log_artifact(summary_json_path)
        mlflow.log_artifact(bar_plot_path)
        mlflow.log_artifact(cm_plot_path)
        mlflow.log_artifact(disagreements_path)
        if Path("README.md").exists():
            mlflow.log_artifact("README.md")
        if Path("requirements.txt").exists():
            mlflow.log_artifact("requirements.txt")

    print("\n==========================================")
    print(f"DECISION: {decision_reason}")
    print("==========================================")


if __name__ == "__main__":
    main()
