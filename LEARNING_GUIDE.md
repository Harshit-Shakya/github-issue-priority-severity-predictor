# 7-Day Learning Plan: GitHub Issue Predictor Project

**Goal**: Understand every component of this project from scratch in 1 week

---

## Day 1: Python & Data Fundamentals (3-4 hours)

### Morning (2 hours)
**Topic**: Python basics + Pandas

**Learn:**
- Python data structures (lists, dicts, tuples)
- Functions and classes
- Pandas DataFrames (read_csv, filtering, groupby)
- NumPy arrays basics

**Resources:**
- [Python Crash Course](https://www.youtube.com/watch?v=rfscVS0vtbw) (2 hours)
- [Pandas in 10 minutes](https://pandas.pydata.org/docs/user_guide/10min.html)

**Practice:**
- Open `01_EDA.ipynb`
- Understand: `pd.read_csv()`, `value_counts()`, `train_test_split()`
- Try: Load a CSV, count values, create splits

### Afternoon (1-2 hours)
**Topic**: Data visualization

**Learn:**
- Matplotlib basics (plt.plot, plt.bar, plt.hist)
- Seaborn for statistical plots

**Resources:**
- [Matplotlib Tutorial](https://www.youtube.com/watch?v=3Xc3CA655Y4) (1 hour)

**Practice:**
- Run cells in `01_EDA.ipynb`
- Modify: Change colors, titles, add your own plots

---

## Day 2: Text Processing & NLP Basics (3-4 hours)

### Morning (2 hours)
**Topic**: Text preprocessing

**Learn:**
- Regular expressions (regex) basics
- String manipulation in Python
- Why clean text data?

**Resources:**
- [Regex Tutorial](https://www.youtube.com/watch?v=sa-TUpSx1JA) (30 min)
- [NLP Text Preprocessing](https://www.youtube.com/watch?v=nxhCyeRR75Q) (1 hour)

**Practice:**
- Open `02_Preprocessing.ipynb`
- Understand: `clean_text()`, `re.sub()`, tokenization
- Try: Write your own text cleaning function

### Afternoon (1-2 hours)
**Topic**: Feature engineering for text

**Learn:**
- What is TF-IDF?
- Bag of words vs embeddings
- Label encoding

**Resources:**
- [TF-IDF Explained](https://www.youtube.com/watch?v=D2V1okCEsiE) (15 min)

**Practice:**
- Run `03_Baseline_Model.ipynb`
- Understand: `TfidfVectorizer`, `fit_transform()`

---

## Day 3: Machine Learning Fundamentals (4-5 hours)

### Morning (2-3 hours)
**Topic**: Classification & evaluation metrics

**Learn:**
- What is classification?
- Logistic Regression intuition
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Class imbalance problem

**Resources:**
- [Classification Explained](https://www.youtube.com/watch?v=yIYKR4sgzI8) (20 min)
- [Precision vs Recall](https://www.youtube.com/watch?v=jJ7ff7Gcq34) (10 min)
- [F1 Score Explained](https://www.youtube.com/watch?v=jJ7ff7Gcq34) (10 min)

**Practice:**
- Run `03_Baseline_Model.ipynb` completely
- Understand: `LogisticRegression()`, `classification_report()`
- Calculate: Precision, recall, F1 manually for one class

### Afternoon (2 hours)
**Topic**: Train/validation/test splits

**Learn:**
- Why split data?
- Overfitting vs underfitting
- Cross-validation basics

**Resources:**
- [Train/Test Split](https://www.youtube.com/watch?v=fwY9Qv96DJY) (15 min)

**Practice:**
- Review `01_EDA.ipynb` splits
- Understand: Why 80/10/10 split?

---

## Day 4: Deep Learning Basics (4-5 hours)

### Morning (2-3 hours)
**Topic**: Neural networks & PyTorch

**Learn:**
- What is a neural network?
- Layers, weights, activation functions
- Forward pass, backward pass (backpropagation)
- PyTorch tensors and basics

**Resources:**
- [Neural Networks Explained](https://www.youtube.com/watch?v=aircAruvnKk) (20 min)
- [PyTorch in 5 minutes](https://www.youtube.com/watch?v=IC0_FRiX-sw) (5 min)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) (1 hour)

**Practice:**
- Create a simple neural network in PyTorch
- Understand: `nn.Module`, `forward()`, `nn.Linear()`

### Afternoon (2 hours)
**Topic**: Training loop & optimization

**Learn:**
- Loss functions (CrossEntropyLoss)
- Optimizers (Adam, SGD)
- Learning rate, epochs, batches
- Gradient descent intuition

**Resources:**
- [Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) (10 min)
- [Adam Optimizer](https://www.youtube.com/watch?v=JXQT_vxqwIs) (10 min)

**Practice:**
- Open `04_DistilBERT_MultiTask.ipynb`
- Understand: `train_epoch()`, `optimizer.step()`, `loss.backward()`

---

## Day 5: Transformers & BERT (4-5 hours)

### Morning (2-3 hours)
**Topic**: Attention mechanism & Transformers

**Learn:**
- What is attention?
- Transformer architecture basics
- BERT and DistilBERT overview
- Tokenization for transformers

**Resources:**
- [Attention Mechanism](https://www.youtube.com/watch?v=fjJOgb-E41w) (15 min)
- [Transformers Explained](https://www.youtube.com/watch?v=SZorAJ4I-sA) (20 min)
- [BERT Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo) (15 min)
- [Hugging Face Transformers](https://huggingface.co/course/chapter1/1) (1 hour)

**Practice:**
- Load DistilBERT tokenizer
- Tokenize sample text
- Understand: `input_ids`, `attention_mask`

### Afternoon (2 hours)
**Topic**: Fine-tuning pre-trained models

**Learn:**
- Transfer learning concept
- Why fine-tune instead of training from scratch?
- Freezing vs unfreezing layers

**Resources:**
- [Transfer Learning](https://www.youtube.com/watch?v=yofjFQddwHE) (15 min)
- [Fine-tuning BERT](https://www.youtube.com/watch?v=x66kkDnbzi4) (20 min)

**Practice:**
- Open `04_DistilBERT_MultiTask.ipynb`
- Understand: `DistilBertModel.from_pretrained()`
- Trace: How text flows through the model

---

## Day 6: Multi-Task Learning & Advanced Concepts (4-5 hours)

### Morning (2-3 hours)
**Topic**: Multi-task learning

**Learn:**
- What is multi-task learning?
- Shared representations
- Multiple loss functions
- When to use multi-task learning?

**Resources:**
- [Multi-Task Learning](https://www.youtube.com/watch?v=UdXfsAr4Gjw) (20 min)

**Practice:**
- Open `04_DistilBERT_MultiTask.ipynb`
- Understand: `MultiTaskDistilBERT` class
- Trace: How one model outputs two predictions
- Understand: `loss = loss_priority + loss_severity`

### Afternoon (2 hours)
**Topic**: Handling class imbalance

**Learn:**
- Class imbalance problem
- Class weights
- Oversampling vs undersampling
- Focal loss

**Resources:**
- [Class Imbalance](https://www.youtube.com/watch?v=X9MZtvvQDR4) (15 min)

**Practice:**
- Understand: `compute_class_weight()` in notebook
- Calculate: Manual class weights for priority
- Understand: Why low priority has weight < 1?

---

## Day 7: Advanced Training & Deployment (4-5 hours)

### Morning (2-3 hours)
**Topic**: Training techniques

**Learn:**
- Gradient clipping (why?)
- Learning rate scheduling
- Early stopping
- Dropout regularization
- Weight decay

**Resources:**
- [Regularization Techniques](https://www.youtube.com/watch?v=6g0t3Phly2M) (20 min)
- [Learning Rate Scheduling](https://www.youtube.com/watch?v=1waHlpKiPJw) (10 min)

**Practice:**
- Open `04_DistilBERT_MultiTask.ipynb`
- Understand: `clip_grad_norm_()`, `scheduler.step()`
- Understand: Early stopping logic
- Modify: Change dropout rate, see what happens

### Afternoon (2 hours)
**Topic**: Model evaluation & deployment

**Learn:**
- Confusion matrix interpretation
- Per-class metrics
- Model saving/loading in PyTorch
- Streamlit basics

**Resources:**
- [Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) (10 min)
- [Streamlit Tutorial](https://www.youtube.com/watch?v=JwSS70SZdyM) (30 min)

**Practice:**
- Run `05_Model_Comparison.ipynb`
- Analyze: Why does "high priority" have low F1?
- Run: `streamlit run app.py`
- Understand: How `app.py` loads and uses the model

---

## Bonus: Interview Prep (Optional)

### Key Questions to Answer:

1. **Why multi-task learning?**
   - Answer: Shared representations, more efficient, related tasks help each other

2. **Why DistilBERT over BERT?**
   - Answer: 40% smaller, 60% faster, 97% of BERT's performance

3. **How did you handle class imbalance?**
   - Answer: Class-weighted loss, gave higher weight to minority classes

4. **Why is F1 score important here?**
   - Answer: Imbalanced data, accuracy is misleading

5. **What's the biggest challenge?**
   - Answer: Severe class imbalance (only 2% high priority)

6. **How would you improve the model?**
   - Answer: Collect more minority class data, try focal loss, ensemble methods

---

## Daily Schedule Template

**Each Day:**
- Morning: 2-3 hours (theory + videos)
- Break: 30 min
- Afternoon: 1-2 hours (hands-on practice)
- Evening: Review notes, run notebooks

**Total Time**: 25-30 hours over 7 days

---

## Key Resources Summary

**Free Courses:**
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - Excellent for intuition
- [Hugging Face NLP Course](https://huggingface.co/course) - Transformers deep dive
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official docs

**YouTube Channels:**
- StatQuest (ML fundamentals)
- 3Blue1Brown (Neural networks)
- Sentdex (Python & ML)

**Practice:**
- Kaggle Learn (free micro-courses)
- Your own notebooks (modify and experiment!)

---

## Progress Checklist

### Day 1
- [ ] Understand Pandas DataFrames
- [ ] Create train/test splits
- [ ] Make basic plots

### Day 2
- [ ] Write regex patterns
- [ ] Clean text data
- [ ] Understand TF-IDF

### Day 3
- [ ] Explain precision, recall, F1
- [ ] Train logistic regression
- [ ] Interpret confusion matrix

### Day 4
- [ ] Build simple neural network
- [ ] Understand forward/backward pass
- [ ] Write training loop

### Day 5
- [ ] Tokenize text with BERT tokenizer
- [ ] Load pre-trained model
- [ ] Understand attention mechanism

### Day 6
- [ ] Explain multi-task learning
- [ ] Calculate class weights
- [ ] Understand shared representations

### Day 7
- [ ] Implement gradient clipping
- [ ] Add early stopping
- [ ] Deploy with Streamlit

---

## Final Project Understanding Test

**Can you answer these?**

1. What does `train_df['text'].apply(lambda x: len(x.split()))` do?
2. Why do we use `max_length=256` in tokenization?
3. What does `torch.argmax(logits, dim=1)` return?
4. Why is "low priority" F1 (0.88) much higher than "high priority" (0.26)?
5. What happens in `loss.backward()`?
6. Why do we use `model.eval()` during inference?
7. What is the purpose of `attention_mask`?
8. How does multi-task learning work in this project?

If you can answer all 8, you understand the project!

---

## Tips for Success

1. **Don't rush** - Understanding > speed
2. **Run every cell** - See outputs, modify code
3. **Break things** - Change parameters, see what breaks
4. **Take notes** - Write down key concepts
5. **Ask questions** - Use ChatGPT/Claude for clarification
6. **Build intuition** - Visualize what's happening
7. **Connect concepts** - How does Day 3 relate to Day 5?

---

## After This Week

**You'll be able to:**
- Explain the entire project in interviews
- Modify the model architecture
- Try different datasets
- Debug training issues
- Answer technical questions confidently

**Next Steps:**
- Try a different dataset
- Implement focal loss
- Add more visualizations
- Write a blog post about the project

Good luck! 🚀
