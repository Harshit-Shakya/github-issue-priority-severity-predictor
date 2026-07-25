# PRD: GitHub Issue Auto-Labeler with Priority Scoring

## Executive Summary

Build an end-to-end NLP system that automatically assigns canonical labels (`bug`,
`feature`, `documentation`, `performance`, `question`, `good-first-issue`, `security`)
to GitHub issues and predicts a priority (LOW / MEDIUM / HIGH). The system handles
noisy real-world issue text — markdown, code blocks, stack traces. Deliver a
reproducible Kaggle notebook pipeline and a Streamlit demo deployed to Hugging Face
Spaces.

**Target:** 4th year B.Tech resume project  
**Timeline:** 5 weeks part-time (~10 hrs/week)  
**Platform:** Kaggle (training, GPU T4) + Hugging Face Spaces (deployment)

---

## Goals & Scope

### Success Criteria (must hit all before calling project done)

| Metric | Target |
|---|---|
| Micro F1 | ≥ 0.75 |
| Macro F1 | ≥ 0.65 |
| Per-label F1 (all labels except `security`) | ≥ 0.50 |
| Per-label F1 (`security`) | ≥ 0.35 (rare label, lower bar acceptable) |
| Hamming Loss | ≤ 0.15 |
| App: prediction latency | < 3 seconds on CPU |

### In Scope

- `lewtun/github-issues` as the sole data source
- Multi-label classification with per-label threshold tuning
- Priority scorer using pseudo-labels
- Streamlit app with text paste mode + GitHub URL fetch mode
- Deployment to Hugging Face Spaces

### Out of Scope

- Label correction / feedback collection UI
- GitHub Actions bot or webhook integration
- Any dataset other than `lewtun/github-issues`

---

## Dataset

**Source:** `lewtun/github-issues` (HuggingFace datasets library)

```python
from datasets import load_dataset
ds = load_dataset("lewtun/github-issues")
# Columns: url, repository_url, title, body, labels, comments
```

**Splits:** 80% train / 10% validation / 10% test. Use stratified splitting on
label co-occurrence where possible. Save exact row indices to `splits.json` at the
start and never regenerate — all downstream notebooks load from this file.

```python
import json
import numpy as np

np.random.seed(42)
indices = np.random.permutation(len(ds["train"]))
n = len(indices)
train_idx = indices[:int(0.8*n)].tolist()
val_idx   = indices[int(0.8*n):int(0.9*n)].tolist()
test_idx  = indices[int(0.9*n):].tolist()

with open("splits.json", "w") as f:
    json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
```

**Label Canonicalization:** Map raw GitHub labels to 7 canonical labels.
Save all mappings to `label_mapping.json`. Log any ambiguous raw labels to
`ambiguous_labels.csv` with a manual decision column — this shows rigor in interviews.

| Canonical Label | Raw GitHub labels that map to it |
|---|---|
| `bug` | bug, defect, error, broken, crash, regression |
| `feature` | feature request, enhancement, new feature, improvement |
| `documentation` | docs, documentation, readme, wiki |
| `performance` | performance, slow, optimization, memory, latency |
| `question` | question, help wanted, support, discussion |
| `good-first-issue` | good first issue, beginner, starter, easy |
| `security` | security, vulnerability, CVE, exploit |

Issues with no label that maps to a canonical label are dropped from the dataset.

---

## Preprocessing Pipeline

All preprocessing lives in `preprocessing.py` and is imported by every notebook.
Never duplicate preprocessing logic across notebooks.

```python
import re

def clean_issue_text(text: str) -> str:
    """
    Replace noisy structures with special tokens.
    Order of operations matters — apply in this exact sequence.
    """
    # 1. Fenced code blocks (multi-line) → [CODE]
    text = re.sub(r"```[\s\S]*?```", " [CODE] ", text)
    # 2. Inline code → [INLINE_CODE]
    text = re.sub(r"`[^`\n]+`", " [INLINE_CODE] ", text)
    # 3. Stack trace lines (Java/Python style) → [STACKTRACE]
    text = re.sub(r"^\s+at\s+\S+\(.*\)$", " [STACKTRACE] ", text, flags=re.MULTILINE)
    text = re.sub(r"Traceback \(most recent call last\)[\s\S]*?(?=\n\n|\Z)",
                  " [STACKTRACE] ", text)
    # 4. URLs → [URL]
    text = re.sub(r"https?://\S+", " [URL] ", text)
    # 5. Markdown images → remove entirely
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    # 6. Markdown links → keep display text only
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # 7. Markdown headers, bold, italic → plain text
    text = re.sub(r"#{1,6}\s", "", text)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    # 8. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input(title: str, body: str) -> str:
    """
    Construct model input. Title is repeated to give it higher weight.
    [STACKTRACE] and [CODE] tokens are preserved as strong label signals.
    """
    cleaned_body = clean_issue_text(body or "")
    clean_title = (title or "").strip()
    return f"[TITLE] {clean_title} [TITLE] {clean_title} [SEP] {cleaned_body}"
```

**Do not lowercase text.** The tokenizer for `distilbert-base-uncased` handles
casing internally. Lowercasing before tokenization is redundant and may hurt
performance on code-heavy text where casing carries meaning.

---

## Model Architecture

### Model 1: Baseline — TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
    ("clf",   OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000,
                                                      class_weight="balanced")))
])
```

Run this first. It will be surprisingly competitive and gives you a justified
reason to use BERT when comparing metrics.

### Model 2: Main — Fine-tuned DistilBERT

Use `distilbert-base-uncased`. Do not switch to a larger model — distilbert fits
within Kaggle's T4 GPU memory, trains in under 2 hours, and is small enough to
load at runtime on Hugging Face Spaces CPU.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class IssueLabeler(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token representation
        return self.classifier(self.dropout(cls))
        # Returns raw logits — sigmoid is applied in loss and inference steps
```

**Loss function:**

```python
# Compute pos_weight per label from training set label frequencies
pos_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - pos_counts
pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Hyperparameters (use these exact values, do not tune further):**

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Warmup steps | 10% of total training steps |
| Max sequence length | 256 tokens |
| Optimizer | AdamW |
| LR scheduler | Linear warmup + linear decay |

**Training loop must:**
- Evaluate on validation set after every epoch
- Save checkpoint only when val Macro F1 improves
- Stop after epoch 4 regardless (no need for early stopping at this scale)
- Set `torch.manual_seed(42)` before training for reproducibility

### Model 3: Priority Scorer (Stage 2)

Priority ground truth does not exist in the dataset. Create pseudo-labels using
the deterministic rule table below. Document this honestly in the README — it is
not a weakness, it is a reasonable design choice for a system where ground truth
is unavailable.

**Pseudo-label rules (apply in order, first match wins):**

| Condition | Priority |
|---|---|
| `security` label predicted | HIGH |
| `bug` + `performance` both predicted | HIGH |
| `bug` predicted alone | MEDIUM |
| `performance` predicted alone | MEDIUM |
| `feature` or `question` predicted | LOW |
| `documentation` or `good-first-issue` predicted | LOW |
| No label predicted | MEDIUM (default) |

**Priority scorer features:**

```python
features = {
    "label_bug":            int("bug" in predicted_labels),
    "label_feature":        int("feature" in predicted_labels),
    "label_documentation":  int("documentation" in predicted_labels),
    "label_performance":    int("performance" in predicted_labels),
    "label_question":       int("question" in predicted_labels),
    "label_gfi":            int("good-first-issue" in predicted_labels),
    "label_security":       int("security" in predicted_labels),
    "conf_max":             float(max(confidence_scores)),
    "conf_mean":            float(mean(confidence_scores)),
    "sentiment_polarity":   float(TextBlob(issue_text).sentiment.polarity),
    "body_length":          int(len(body.split())),
    "has_stacktrace":       int("[STACKTRACE]" in cleaned_text),
    "has_code":             int("[CODE]" in cleaned_text),
}
```

Model: `XGBoost` classifier with `objective="multi:softmax"`, `num_class=3`.
Evaluate using precision / recall / F1 for the HIGH class specifically — that is
the class that matters most in a real system.

---

## Per-Label Threshold Tuning

Run this on the validation set after training. Never tune thresholds on the test set.

```python
import numpy as np
from sklearn.metrics import f1_score

def tune_thresholds(val_probs: np.ndarray, val_labels: np.ndarray,
                    label_names: list) -> dict:
    thresholds = {}
    for i, name in enumerate(label_names):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.10, 0.90, 0.02):
            preds = (val_probs[:, i] > t).astype(int)
            f1 = f1_score(val_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[name] = round(float(best_t), 2)
    return thresholds

# Save immediately after tuning — this file is used by inference.py and the app
import json
with open("thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)
```

**Inference rule:** If no label exceeds its tuned threshold, set `needs_review=True`
and do not force a prediction. This is a first-class output of the system, not an
error state.

---

## Evaluation

Report all of the following. Do not report accuracy — it is meaningless for
multi-label classification.

```python
from sklearn.metrics import classification_report, hamming_loss

# Per-label F1, precision, recall
print(classification_report(y_test, y_pred, target_names=label_names))

# Hamming loss
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")

# Micro and Macro F1
from sklearn.metrics import f1_score
print(f"Micro F1: {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
```

Also generate:
- PR curve per label (use `sklearn.metrics.precision_recall_curve`)
- Label co-occurrence heatmap on test set predictions vs ground truth

---

## Notebook Pipeline

Run in this exact order. Each notebook saves its outputs to disk so the next
notebook can load them without re-running earlier steps.

```
notebooks/
├── 01_eda.ipynb
│     Load dataset → label distribution plots → text length histograms
│     → label co-occurrence heatmap → save splits.json + label_mapping.json
│
├── 02_preprocessing.ipynb
│     Load splits.json → apply clean_issue_text() + build_input()
│     → save processed_issues.csv
│
├── 03_baseline.ipynb
│     Load processed_issues.csv → TF-IDF + LR → report metrics
│     → save tfidf_lr_model.pkl
│
├── 04_bert_finetune.ipynb       ← run with Kaggle GPU T4 accelerator
│     Load processed_issues.csv → fine-tune IssueLabeler → tune thresholds
│     → save bert_labeler/ (HF format) + thresholds.json
│
├── 05_priority_scorer.ipynb
│     Load bert_labeler/ → generate pseudo-labels → train XGBoost
│     → save priority_model.pkl
│
└── 06_model_comparison.ipynb
      Load all models → side-by-side metrics table → PR curves
      → per-label F1 bar chart → SHAP on TF-IDF model
```

---

## Project Structure

```
github-issue-labeler/
├── data/
│   ├── raw/                          # original HF dataset (cached)
│   └── processed/
│       ├── processed_issues.csv
│       ├── splits.json
│       ├── label_mapping.json
│       └── ambiguous_labels.csv
├── models/
│   ├── tfidf_lr_model.pkl
│   ├── bert_labeler/                 # saved in HuggingFace format
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer/
│   ├── thresholds.json
│   └── priority_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_bert_finetune.ipynb
│   ├── 05_priority_scorer.ipynb
│   └── 06_model_comparison.ipynb
├── preprocessing.py                  # shared — imported by all notebooks + app
├── inference.py                      # load models + predict (used by app.py)
├── app.py                            # Streamlit app
├── requirements.txt
└── README.md
```

---

## Streamlit App

### Page 1: Paste Issue Text

- Text input: Issue title
- Text area: Issue body
- Button: "Predict"
- Output: label badges with confidence bars (one per label)
- Output: priority badge (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)
- Output: "Needs human review" warning banner when no label clears threshold
- Expandable panel: "Why this label?" — top 5 TF-IDF words per predicted label
  (use SHAP values computed on the baseline model; fast enough for real-time)

### Page 2: GitHub URL Mode

- Input: paste any public GitHub issue URL in format
  `https://github.com/{owner}/{repo}/issues/{number}`
- Parse owner, repo, number from URL using regex
- Fetch via unauthenticated GitHub public API:
  ```python
  url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
  r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
  ```
- Show a clear error message if: issue is private, URL is malformed, or rate limit
  is reached (HTTP 403 / 404)
- On success: same output as Page 1

### Page 3: Model Explorer

- Side-by-side metrics table: Baseline vs DistilBERT
- Per-label F1 bar chart (interactive, using Plotly)
- Threshold slider: let user drag threshold for one label and see live how
  predictions on a fixed example change — demonstrates threshold tuning concept

---

## Tech Stack

| Component | Tool |
|---|---|
| Data loading | HuggingFace `datasets` |
| Text cleaning | Python `re`, `preprocessing.py` |
| Baseline model | scikit-learn (TF-IDF + OneVsRest LR) |
| Explainability | SHAP (on baseline model) |
| Sentiment | TextBlob |
| Deep learning | PyTorch + HuggingFace Transformers |
| Priority model | XGBoost |
| Experiment tracking | Weights & Biases (free tier) — log all training runs |
| App | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Deployment

**Do not bundle BERT weights in the repo.** Load from HuggingFace Hub at runtime:

```python
# In inference.py — load model from HF Hub instead of local path
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("your-hf-username/github-issue-labeler")
tokenizer = AutoTokenizer.from_pretrained("your-hf-username/github-issue-labeler")
```

Push your fine-tuned model to HF Hub first using `model.push_to_hub(...)`.
This also means your model is publicly accessible, which is good for the resume.

**requirements.txt (pinned):**
```
torch==2.1.0
transformers==4.36.2
datasets==2.14.6
scikit-learn==1.3.2
xgboost==2.0.3
shap==0.43.0
streamlit==1.28.2
textblob==0.17.1
plotly==5.17.0
requests==2.31.0
```

---

## Week-by-Week Milestones

### Week 1 — Data & EDA
- Load `lewtun/github-issues`, explore label distribution
- Implement label canonicalization, save `label_mapping.json`
- Generate `splits.json` with fixed seed
- Run EDA: label frequency, co-occurrence heatmap, text length distribution
- Deliverable: `01_eda.ipynb` complete, splits and mappings saved

### Week 2 — Preprocessing & Baseline
- Implement `clean_issue_text()` and `build_input()` in `preprocessing.py`
- Apply to full dataset, save `processed_issues.csv`
- Train TF-IDF + LR baseline, report all metrics
- Deliverable: `02_preprocessing.ipynb` + `03_baseline.ipynb` complete

### Week 3 — BERT Fine-tuning + Threshold Tuning
- Fine-tune `distilbert-base-uncased` on Kaggle GPU
- Run per-label threshold tuning on validation set
- Save `bert_labeler/` and `thresholds.json`
- Deliverable: `04_bert_finetune.ipynb` complete, model weights saved

### Week 4 — Priority Scorer + Model Comparison
- Generate pseudo-labels, train XGBoost priority scorer
- Run `06_model_comparison.ipynb` — produce final metrics table and charts
- Push fine-tuned model to HuggingFace Hub
- Deliverable: all notebooks complete, models saved and pushed

### Week 5 — Streamlit App + Deployment
- Build `app.py` with all 3 pages
- Test locally end-to-end with both text paste and URL input
- Deploy to Hugging Face Spaces
- Write README with metrics table, app link, and architecture diagram
- Deliverable: public app URL, clean GitHub repo, resume bullets filled in

---

## Reproducibility Checklist

Before submitting / sharing the project, verify all of the following:

- [ ] `splits.json` exists and is committed to the repo
- [ ] `label_mapping.json` exists and is committed to the repo
- [ ] `thresholds.json` exists and is committed to the repo
- [ ] All notebooks load `splits.json` — no notebook regenerates splits on its own
- [ ] `random_seed = 42` is set at the top of every training notebook
- [ ] Running notebooks 01 → 06 in order produces the same metrics as reported
- [ ] `requirements.txt` has pinned versions for all packages
- [ ] Fine-tuned model is pushed to HuggingFace Hub and loadable with one line

---

## Key Interview Questions & Answers

**"Why sigmoid and not softmax?"**
Softmax forces all label probabilities to sum to 1, implying the labels are mutually
exclusive. Here they are not — an issue can be both `bug` and `performance` at the
same time. Sigmoid treats each label as an independent binary decision.

**"How did you handle class imbalance?"**
Two ways: (1) `pos_weight` in `BCEWithLogitsLoss` — each label's positive samples
get a weight equal to neg/pos ratio so rare labels aren't ignored during training.
(2) Per-label threshold tuning on the validation set — rare labels like `security`
get a lower threshold so the model is more willing to predict them.

**"Why per-label thresholds instead of a fixed 0.5?"**
Different labels have different base rates. A fixed threshold of 0.5 works fine for
common labels but causes the model to almost never predict rare labels. Tuning each
label's threshold independently on the validation set to maximize per-label F1
directly optimizes the metric we care about.

**"What happens when the model isn't confident?"**
If no label exceeds its tuned threshold, the issue is flagged as "needs human
review" — the system doesn't force a wrong prediction. This is standard practice
in production ML systems.

**"Where did the priority labels come from? You said there's no ground truth."**
I created pseudo-labels using a deterministic rule table based on label combinations
(e.g., `security` → HIGH, `bug` + `performance` → HIGH). I documented the rules
explicitly and mentioned their limitations in the README. In production you'd
collect ground truth from engineering team triage data over time.

**"How would you handle a new label appearing?"**
Add a new column to the label matrix, collect training examples for it, and retrain
only the classification head while keeping the DistilBERT encoder frozen. This is
much faster than full fine-tuning. As a stopgap before enough data is collected,
you could use zero-shot classification with a model like `facebook/bart-large-mnli`.

---

## Resume Bullets (fill in [X] and [Y] after training)

- Built a 7-label multi-label GitHub issue classifier using fine-tuned DistilBERT with per-label threshold tuning; achieved Micro F1 = [X], Macro F1 = [Y] on held-out test set
- Designed a two-stage NLP pipeline: multi-label issue labeler → XGBoost priority scorer (LOW/MEDIUM/HIGH) using predicted labels, confidence scores, and sentiment features
- Engineered a noise-robust text preprocessing pipeline handling code blocks, stack traces, markdown, and URLs from raw GitHub issue text; implemented as a shared module across all training notebooks
- Deployed an interactive Streamlit app to Hugging Face Spaces with live GitHub URL fetching, adjustable per-label confidence thresholds, and a model comparison explorer
