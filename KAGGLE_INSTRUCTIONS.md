# How to Run DistilBERT on Kaggle (FREE GPU)

## Step 1: Run Notebooks 1-3 Locally

```bash
# Open Jupyter
jupyter notebook

# Run in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Preprocessing.ipynb  
# 3. notebooks/03_Baseline_Model.ipynb
```

This will create:
- `artifacts/splits.json`
- `artifacts/label_mappings.json`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

---

## Step 2: Prepare Files for Kaggle

After running notebooks 1-3, you'll need these files:
1. `data/processed/train.csv`
2. `data/processed/val.csv`
3. `data/processed/test.csv`
4. `notebooks/04_DistilBERT_MultiTask.ipynb`

---

## Step 3: Upload to Kaggle

### A. Create Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up / Log in

### B. Create a New Dataset
1. Click your profile → "Datasets" → "New Dataset"
2. Upload these 3 files:
   - `train.csv`
   - `val.csv`
   - `test.csv`
3. Name it: "github-issues-processed"
4. Make it Private
5. Click "Create"

### C. Create a New Notebook
1. Click "Code" → "New Notebook"
2. Click "File" → "Upload Notebook"
3. Upload `04_DistilBERT_MultiTask.ipynb`

### D. Add Your Dataset
1. In the notebook, click "Add Data" (right sidebar)
2. Search for your dataset "github-issues-processed"
3. Click "Add"

### E. Enable GPU
1. Click "Settings" (right sidebar)
2. Under "Accelerator", select "GPU T4 x2" or "GPU P100"
3. Click "Save"

### F. Modify Paths in Notebook
Change these lines in the notebook:

```python
# OLD:
train_df = pd.read_csv('../data/processed/train.csv')
val_df = pd.read_csv('../data/processed/val.csv')
test_df = pd.read_csv('../data/processed/test.csv')

# NEW:
train_df = pd.read_csv('/kaggle/input/github-issues-processed/train.csv')
val_df = pd.read_csv('/kaggle/input/github-issues-processed/val.csv')
test_df = pd.read_csv('/kaggle/input/github-issues-processed/test.csv')
```

And change save path:

```python
# OLD:
model.save('../models/distilbert_multitask')

# NEW:
model.save('/kaggle/working/distilbert_multitask')
```

---

## Step 4: Run the Notebook

1. Click "Run All" or run cells one by one
2. Wait ~1-2 hours for training
3. Monitor progress in output

---

## Step 5: Download Trained Model

After training completes:
1. Click "Output" tab (top right)
2. Download `distilbert_multitask` folder
3. Move it to your local `models/` folder

---

## Alternative: Use Kaggle CLI (Advanced)

If you're comfortable with command line:

```bash
# Install Kaggle CLI
pip install kaggle

# Get API token from kaggle.com/settings
# Place kaggle.json in ~/.kaggle/

# Upload dataset
kaggle datasets create -p data/processed

# Upload and run notebook
kaggle kernels push -p notebooks/
```

---

## Troubleshooting

**"Out of Memory" error:**
- Reduce BATCH_SIZE from 16 to 8
- Reduce MAX_LEN from 256 to 128

**"Session timeout":**
- Kaggle has 9-hour limit
- Your training should finish in 1-2 hours

**"GPU not available":**
- Make sure you selected GPU in Settings
- Try refreshing the page

---

## Expected Results

After training on Kaggle GPU:
- **Training time**: 1-2 hours
- **Priority F1**: ~0.85-0.90
- **Severity F1**: ~0.75-0.85
- **Model size**: ~250 MB

---

## Quick Summary

1. ✅ Run notebooks 1-3 locally (30-45 min)
2. ✅ Upload train/val/test CSVs to Kaggle as dataset
3. ✅ Upload notebook 04 to Kaggle
4. ✅ Enable GPU
5. ✅ Fix file paths
6. ✅ Run notebook (1-2 hours)
7. ✅ Download trained model

**Total time: ~2-3 hours**
