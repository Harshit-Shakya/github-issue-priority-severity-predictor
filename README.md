# GitHub Issue Priority & Severity Predictor

Multi-task learning model to automatically predict priority and severity for GitHub issues using DistilBERT.

## Overview

This project uses a fine-tuned DistilBERT model with multi-task learning to simultaneously predict:

- **Priority**: low, medium, high
- **Severity**: Minor, Major, Critical

## Dataset

- **Source**: 114,073 GitHub issues from 68 major repositories (PyTorch, Flutter, Rust, Go, VS Code, etc.)
- **Split**: 80% train, 10% validation, 10% test
- **Features**: Issue title + body text

## Model Architecture

- **Base Model**: DistilBERT (66M parameters)
- **Approach**: Multi-task learning with shared representations
- **Heads**: 2 task-specific classification heads with hidden layers
- **Training**: 3 epochs with early stopping, class-weighted loss, gradient clipping

## Results

| Metric      | Baseline (TF-IDF) | DistilBERT | Improvement |
| ----------- | ----------------- | ---------- | ----------- |
| Priority F1 | 0.65              | 0.81       | +24.6%      |
| Severity F1 | 0.60              | 0.79       | +31.7%      |
| Average F1  | 0.625             | 0.80       | +28.0%      |

### Per-Class Performance

**Priority:**

- Low: 0.88 F1 (89% of data)
- Medium: 0.29 F1 (9% of data)
- High: 0.26 F1 (2% of data)

**Severity:**

- Critical: 0.91 F1 (58% of data)
- Minor: 0.68 F1 (26% of data)
- Major: 0.51 F1 (16% of data)

## Project Structure

```
.
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb          # Text cleaning & encoding
│   ├── 03_Baseline_Model.ipynb         # TF-IDF + LogReg baseline
│   ├── 04_DistilBERT_MultiTask.ipynb   # Main DistilBERT model
│   └── 05_Model_Comparison.ipynb       # Visualizations & analysis
├── app.py                               # Streamlit demo app
├── requirements.txt                     # Python dependencies
└── README.md
```

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd github-issue-predictor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Run notebooks in order:

1. `01_EDA.ipynb` - Create train/val/test splits
2. `02_Preprocessing.ipynb` - Clean and encode text
3. `03_Baseline_Model.ipynb` - Train baseline (optional)
4. `04_DistilBERT_MultiTask.ipynb` - Train DistilBERT model

### Inference (Streamlit App)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Programmatic Usage

```python
import torch
from transformers import DistilBertTokenizer

# Load model
model = MultiTaskDistilBERT()
model.load_state_dict(torch.load('distilbert_multitask_final.pth'))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Predict
title = "Application crashes on startup"
body = "The app crashes immediately after launching..."
text = f"[TITLE] {title} [TITLE] {title} [SEP] {body}"

encoding = tokenizer(text, max_length=256, truncation=True,
                     padding='max_length', return_tensors='pt')

with torch.no_grad():
    priority_logits, severity_logits = model(
        encoding['input_ids'],
        encoding['attention_mask']
    )

priority = ['low', 'medium', 'high'][torch.argmax(priority_logits)]
severity = ['Minor', 'Major', 'Critical'][torch.argmax(severity_logits)]

print(f"Priority: {priority}, Severity: {severity}")
```

## Key Features

- **Multi-task learning**: Single model predicts both priority and severity
- **Class imbalance handling**: Class-weighted loss functions
- **Regularization**: Dropout (0.3), weight decay (0.01), gradient clipping
- **Early stopping**: Prevents overfitting with patience=2
- **Production-ready**: 80% weighted F1 score

## Technologies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for DistilBERT
- **scikit-learn**: Baseline models and metrics
- **Streamlit**: Interactive web app
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualizations

## Limitations

- Performance degrades on minority classes (medium/high priority) due to severe class imbalance
- Model trained on specific repositories may not generalize to all GitHub projects
- Requires GPU for reasonable training time (~1.5 hours on Kaggle GPU)

## Future Improvements

- Collect more data for minority classes
- Try focal loss or oversampling techniques
- Experiment with larger models (RoBERTa, BERT-large)
- Add repository-specific fine-tuning
- Implement active learning for continuous improvement

## License

MIT

## Author

Harshit Shakya

## Acknowledgments

- Dataset: GitHub Issues from 68 major open-source repositories
- Base Model: DistilBERT by Hugging Face
- Trained on Kaggle GPU
