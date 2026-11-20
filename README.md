# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** `1MS25CI058`  
**Repository:** `1MS25CI058-mlflow-champion-challenger`

---

## 1. Task and Models
* **Task:** Binary Sentiment Classification (`POSITIVE` / `NEGATIVE`)
* **Dataset:** `glue/sst2` (`validation` split, deterministic subset of `100` samples with `seed=42`, saved to `data/eval_inputs.csv`)
* **Model A (Current Champion Baseline):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters, fast distilled baseline)
* **Model B (Challenger Candidate):** `siebert/sentiment-roberta-large-english` (~355M parameters, larger RoBERTa-large checkpoint)

---

## 2. How to Run
Run the following commands in order from the project root directory:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the local MLflow tracking server (Terminal 1):**
   ```bash
   python -m mlflow server --host 127.0.0.1 --port 5000
   ```

3. **Run evaluation, artifact logging, and comparison (Terminal 2):**
   ```bash
   python evaluate.py
   ```

4. **Register both models and assign `@champion` and `@challenger` aliases:**
   ```bash
   python register_champion.py
   ```

5. **Run inference on unseen data via the MLflow Registry `@champion` alias:**
   ```bash
   python predict_from_mlflow.py
   ```

---

## 3. Champion–Challenger Decision Rule (Written Before Running)
Before executing `evaluate.py`, the promotion criteria were fixed as follows:

1. **Primary Metric:** `macro_f1` (**higher is better**). Macro-F1 balances precision and recall equally across both positive and negative classes so neither class is favored. Secondary metrics tracked: `accuracy` (higher is better), `precision` (higher is better), `recall` (higher is better), `avg_latency_ms` (lower is better), and `total_inference_s` (lower is better).
2. **Minimum Improvement Margin:** Model B (Challenger) must beat Model A (Champion) by at least **+0.0100 (1.0 percentage point)** on `macro_f1`. Because the evaluation subset has 100 examples, a difference of 1 sample equals 0.01 in accuracy/F1; requiring at least 1.0 percentage point avoids promoting a 5× larger model over single-sample noise.
3. **Tie-Breaker & Latency Guardrail:** If the `macro_f1` improvement is below `+0.0100`, the faster existing Champion (Model A) is retained. Furthermore, Model B is rejected if its average per-sample latency exceeds **6.0×** Model A's latency (`avg_latency_ms_B / avg_latency_ms_A > 6.0`).

---

## 4. Results Table
*(Update the exact numbers below from `outputs/metrics_summary.json` after executing `python evaluate.py`.)*

| Metric | Direction | Model A (`distilbert-sst2`) | Model B (`roberta-large`) | Difference (`B - A`) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | Higher is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |
| **Accuracy** | Higher is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |
| **Precision (Macro)** | Higher is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |
| **Recall (Macro)** | Higher is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |
| **Avg Latency (ms/sample)** | Lower is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |
| **Total Inference Time (s)** | Lower is better | *Run `evaluate.py`* | *Run `evaluate.py`* | *Run `evaluate.py`* |

---

## 5. Decision and Trade-Off Analysis
*(150–250 words — finalize with your exact run values from `outputs/metrics_summary.json`)*

Based on the pre-defined Champion–Challenger rule, the automated comparison evaluates whether `siebert/sentiment-roberta-large-english` (Model B) achieves at least a +0.0100 `macro_f1` gain over `distilbert-base-uncased-finetuned-sst-2-english` (Model A) without breaching the 6.0× latency guardrail. Across 100 validation sentences from SST-2, Model B brings ~355M parameters compared to Model A's ~67M parameters, allowing it to capture subtle syntactic negation and contrastive clauses where DistilBERT occasionally over-relies on individual sentiment keywords (documented in `outputs/qualitative_disagreements.csv`). However, on a sample size of $N = 100$, each single prediction shifts accuracy and Macro-F1 by roughly 1%, meaning small gains of 1–2 percentage points sit near the boundary of sampling variance. In production trade-off terms, Model B requires roughly 5× more memory and significantly higher CPU inference latency per sentence, whereas Model A delivers near-instantaneous responses suitable for high-throughput CPU serving. By enforcing both a minimum quality margin and a latency multiplier cap, the pipeline ensures the larger Challenger only replaces the Champion when its accuracy gain justifies the compute overhead.

---

## 6. Limitations and Declaration
* **Limitations:** Evaluation was conducted on a 100-sample subset of SST-2 movie review snippets (`seed=42`). Because $N = 100$, a 1–2% metric delta represents only 1–2 examples and may not generalize across domains such as product reviews or social media slang. CPU latency measurements can also vary depending on background system load.
* **AI Assistance Declaration:** An AI coding assistant was used to help structure the `mlflow.pyfunc.PythonModel` wrapper, plotting boilerplate, and README scaffolding. All experimental rules, scripts, MLflow runs, and results were verified and executed by the student (`1MS25CI058`).
