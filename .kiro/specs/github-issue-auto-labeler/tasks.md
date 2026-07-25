# Implementation Plan: GitHub Issue Auto-Labeler

## Overview

This implementation plan breaks down the GitHub Issue Auto-Labeler system into discrete, incremental coding tasks. The approach follows a bottom-up strategy: build core data infrastructure first, then preprocessing, then models (baseline → main → priority), then application interfaces, and finally deployment. Each task builds on previous work, with property-based tests integrated throughout to validate correctness early.

The implementation uses Python with TensorFlow/Keras for deep learning, scikit-learn for baseline models, XGBoost for priority scoring, and Streamlit for the web interface.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/unit/`, `tests/property/`, `notebooks/`, `streamlit_app/`, `artifacts/`
  - Create `requirements.txt` with all dependencies: datasets, transformers, tensorflow, scikit-learn, xgboost, streamlit, plotly, shap, hypothesis, pytest, textblob
  - Create `setup.py` for package installation
  - Initialize git repository and create `.gitignore` for Python projects
  - _Requirements: 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8_

- [ ] 2. Implement Dataset Manager
  - [x] 2.1 Create `src/data/dataset_manager.py` with DatasetManager class
    - Implement `load_dataset()` to fetch lewtun/github-issues from HuggingFace
    - Implement `create_splits()` with 80/10/10 ratio and seed=42
    - Implement `save_splits()` and `load_splits()` for splits.json
    - Implement `create_label_mapping()` for 7 canonical labels
    - Implement `save_label_mapping()` and `load_label_mapping()` for label_mapping.json
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for dataset split ratios
    - **Property 1: Dataset Split Ratios**
    - **Validates: Requirements 1.2**
    - Generate random dataset sizes (100-10000), verify 80/10/10 splits within ±2% tolerance
    - _Requirements: 1.2_

  - [ ]* 2.3 Write property tests for serialization round-trips
    - **Property 2: Split Serialization Round-Trip**
    - **Property 3: Label Mapping Serialization Round-Trip**
    - **Validates: Requirements 1.3, 1.5**
    - Test that save/load operations preserve data integrity
    - _Requirements: 1.3, 1.5_

  - [ ]* 2.4 Write unit tests for Dataset Manager
    - Test label mapping contains exactly 7 labels with correct names
    - Test splits.json and label_mapping.json file creation
    - Test error handling for invalid dataset sizes
    - _Requirements: 1.1, 1.4_

- [ ] 3. Implement Text Preprocessor
  - [x] 3.1 Create `src/preprocessing/text_preprocessor.py` with TextPreprocessor class
    - Implement `remove_markdown()` to strip markdown syntax (**, __, ##, etc.)
    - Implement `normalize_code_blocks()` to extract and normalize ```...``` blocks
    - Implement `normalize_stack_traces()` to standardize stack trace formatting
    - Implement `replace_urls()` to replace URLs with [URL] token
    - Implement `preprocess()` to orchestrate all preprocessing steps
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 3.2 Write property test for preprocessing completeness
    - **Property 4: Text Preprocessing Completeness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    - Generate random text with markdown, code blocks, stack traces, URLs
    - Verify all elements are removed/normalized
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 3.3 Write property test for preprocessing idempotence
    - **Property 5: Preprocessing Idempotence**
    - **Validates: Requirements 2.6**
    - Verify preprocess(preprocess(text)) == preprocess(text) for all inputs
    - _Requirements: 2.6_

  - [ ]* 3.4 Write unit tests for Text Preprocessor
    - Test markdown removal with specific examples
    - Test code block extraction with various formats (python, javascript, etc.)
    - Test URL replacement with different URL patterns
    - Test edge cases: empty input, very long input (10000+ chars), special characters
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Create Kaggle notebook for EDA
  - [x] 4.1 Create `notebooks/01_exploratory_data_analysis.ipynb`
    - Load dataset using DatasetManager
    - Analyze label distribution across canonical labels
    - Visualize issue length distribution (title + body)
    - Analyze label co-occurrence patterns
    - Explore text characteristics (markdown usage, code blocks, URLs)
    - Create visualizations with Plotly
    - _Requirements: 9.1_

- [ ] 5. Create Kaggle notebook for preprocessing
  - [ ] 5.1 Create `notebooks/02_data_preprocessing.ipynb`
    - Load dataset and splits using DatasetManager
    - Apply TextPreprocessor to all splits
    - Show before/after examples of preprocessing
    - Analyze preprocessing impact on text length
    - Save preprocessed data for model training
    - _Requirements: 9.2_

- [ ] 6. Checkpoint - Verify data pipeline
  - Ensure all tests pass for Dataset Manager and Text Preprocessor
  - Verify splits.json and label_mapping.json are generated correctly
  - Ask the user if questions arise

- [ ] 7. Implement Baseline Model
  - [ ] 7.1 Create `src/models/baseline_model.py` with BaselineModel class
    - Initialize TfidfVectorizer with max_features=10000, ngram_range=(1,2)
    - Initialize LogisticRegression with C=1.0, max_iter=1000, class_weight='balanced'
    - Use MultiOutputClassifier for multi-label classification
    - Implement `fit()` to train on preprocessed text and multi-label targets
    - Implement `predict_proba()` to return 7-dimensional probability vector
    - Implement `save()` and `load()` for model persistence
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 7.2 Write property tests for baseline model output
    - **Property 6: Model Output Dimensionality**
    - **Property 7: Multi-Label Classification Support**
    - **Validates: Requirements 3.4, 3.5**
    - Verify output is 7-dimensional for all inputs
    - Verify probabilities are independent (can sum > 1.0)
    - _Requirements: 3.4, 3.5_

  - [ ]* 7.3 Write unit tests for Baseline Model
    - Test model initialization with correct components (TF-IDF, LogReg)
    - Test training with small synthetic dataset
    - Test prediction output format and value ranges [0, 1]
    - Test save/load functionality
    - _Requirements: 3.1, 3.2, 3.4_

- [ ] 8. Create Kaggle notebook for baseline model
  - [ ] 8.1 Create `notebooks/03_baseline_model.ipynb`
    - Load preprocessed data from splits
    - Train BaselineModel on training split
    - Evaluate on validation and test splits
    - Calculate Micro F1, Macro F1, Hamming Loss, per-label F1
    - Create performance visualizations with Plotly
    - Save trained baseline model
    - _Requirements: 9.3_

- [ ] 9. Implement Main Model (DistilBERT)
  - [ ] 9.1 Create `src/models/distilbert_model.py` with DistilBERTClassifier class
    - Initialize DistilBERT base model using TensorFlow/Keras and transformers library
    - Add classification head: Dense(768 -> 7) with sigmoid activation
    - Implement `compile_model()` with Adam optimizer (lr=2e-5) and binary cross-entropy loss
    - Implement tokenization with max_length=512
    - Implement `fit()` with training and validation data, early stopping
    - Implement `predict_proba()` to return 7-dimensional probability vector
    - Implement `save()` and `load()` for model weights
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 9.2 Write property test for DistilBERT output
    - **Property 6: Model Output Dimensionality**
    - **Property 7: Multi-Label Classification Support**
    - **Validates: Requirements 4.4, 4.5**
    - Verify output is 7-dimensional for all inputs
    - Verify multi-label capability
    - _Requirements: 4.4, 4.5_

  - [ ]* 9.3 Write property test for model weight serialization
    - **Property 8: Model Weight Serialization Round-Trip**
    - **Validates: Requirements 4.6**
    - Train small model, save weights, load weights, verify predictions match (within 1e-6)
    - _Requirements: 4.6_

  - [ ]* 9.4 Write property test for inference latency
    - **Property 9: Inference Latency Constraint**
    - **Validates: Requirements 5.6**
    - Generate random inputs ≤ 512 tokens, verify prediction time < 3 seconds on CPU
    - _Requirements: 5.6_

  - [ ]* 9.5 Write unit tests for DistilBERT Model
    - Test model architecture initialization with TensorFlow/Keras
    - Test tokenization and input preparation
    - Test training with small synthetic dataset (1 epoch)
    - Test prediction output format
    - _Requirements: 4.1, 4.2, 4.4_

- [ ] 10. Create Kaggle notebook for DistilBERT fine-tuning
  - [ ] 10.1 Create `notebooks/04_distilbert_finetuning.ipynb`
    - Load preprocessed data from splits
    - Initialize DistilBERTClassifier
    - Fine-tune on training split with validation monitoring (3-5 epochs)
    - Evaluate on validation and test splits
    - Calculate Micro F1, Macro F1, Hamming Loss, per-label F1
    - Compare performance with baseline model
    - Save trained DistilBERT model
    - _Requirements: 9.4_

- [ ] 11. Checkpoint - Verify model pipeline
  - Ensure all tests pass for Baseline and DistilBERT models
  - Verify performance targets are achievable (Micro F1 ≥ 0.75, Macro F1 ≥ 0.65)
  - Ask the user if questions arise

- [ ] 12. Implement Threshold Tuner
  - [ ] 12.1 Create `src/models/threshold_tuner.py` with ThresholdTuner class
    - Implement `optimize_thresholds()` to find optimal threshold per label
    - For each label: try thresholds 0.1 to 0.9 in steps of 0.05, maximize F1 score
    - Implement `save_thresholds()` and `load_thresholds()` for thresholds.json
    - Implement `apply_thresholds()` to convert probabilities to binary predictions
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [ ]* 12.2 Write property test for threshold serialization
    - **Property 10: Threshold Serialization Round-Trip**
    - **Validates: Requirements 6.3**
    - Verify save/load preserves threshold values
    - _Requirements: 6.3_

  - [ ]* 12.3 Write property test for threshold application
    - **Property 11: Threshold Application**
    - **Validates: Requirements 6.5**
    - Generate random probabilities and thresholds, verify only labels above threshold are included
    - _Requirements: 6.5_

  - [ ]* 12.4 Write unit tests for Threshold Tuner
    - Test threshold optimization with known probabilities and labels
    - Test threshold application with edge cases (all above, all below, mixed)
    - Test thresholds.json file creation and loading
    - _Requirements: 6.1, 6.3, 6.5_

- [ ] 13. Implement Label Classifier orchestrator
  - [ ] 13.1 Create `src/models/label_classifier.py` with LabelClassifier class
    - Initialize with model (Baseline or DistilBERT), preprocessor, and thresholds
    - Implement `predict()` to orchestrate: preprocess → model → apply thresholds → needs_review
    - Calculate needs_review flag: True if no label exceeds threshold, False otherwise
    - Return dict with labels, probabilities, needs_review, label_vector
    - _Requirements: 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 13.2 Write property test for needs_review flag logic
    - **Property 12: Needs Review Flag Logic**
    - **Validates: Requirements 7.2, 7.3**
    - Generate random probabilities and thresholds, verify flag is correct
    - _Requirements: 7.2, 7.3_

  - [ ]* 13.3 Write unit tests for Label Classifier
    - Test needs_review flag with specific threshold scenarios
    - Test integration with preprocessor and model (use mocked model)
    - Test output format includes all required fields
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 14. Implement Priority Scorer
  - [ ] 14.1 Create `src/models/priority_scorer.py` with PriorityScorer class
    - Implement `generate_pseudo_labels()` with deterministic rules:
      - security → HIGH (2)
      - bug + performance → HIGH (2)
      - good-first-issue → LOW (0)
      - question → LOW (0)
      - bug or performance → MEDIUM (1)
      - feature → MEDIUM (1)
      - default → LOW (0)
    - Initialize XGBoost classifier with max_depth=5, learning_rate=0.1, n_estimators=100
    - Implement `fit()` to train on 7-dimensional label vectors and pseudo-labels
    - Implement `predict()` and `predict_proba()` for priority prediction
    - Implement `save()` and `load()` for model persistence
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [ ]* 14.2 Write property test for priority output validity
    - **Property 13: Priority Level Output Validity**
    - **Validates: Requirements 8.2**
    - Generate random label vectors, verify output is one of {0, 1, 2}
    - _Requirements: 8.2_

  - [ ]* 14.3 Write property test for pseudo-label generation rules
    - **Property 14: Pseudo-Label Generation Rules**
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7**
    - Generate label vectors with specific combinations, verify correct priority assigned
    - Test: security=1 → HIGH, bug=1 & performance=1 → HIGH, good-first-issue=1 → LOW, question=1 → LOW
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 14.4 Write unit tests for Priority Scorer
    - Test pseudo-label generation with specific label combinations
    - Test XGBoost model training and prediction with synthetic data
    - Test edge cases: all labels false, all labels true
    - Test save/load functionality
    - _Requirements: 8.1, 8.2, 8.3, 8.8_

- [ ] 15. Create Kaggle notebook for priority scorer
  - [ ] 15.1 Create `notebooks/05_priority_scorer.ipynb`
    - Load trained label classifier (DistilBERT)
    - Generate predictions on training data to get label vectors
    - Generate pseudo-labels using deterministic rules
    - Train PriorityScorer on label vectors and pseudo-labels
    - Evaluate on validation and test sets
    - Create confusion matrix and feature importance visualizations
    - Save trained priority scorer
    - _Requirements: 9.5_

- [ ] 16. Checkpoint - Verify complete model pipeline
  - Ensure all tests pass for Threshold Tuner, Label Classifier, and Priority Scorer
  - Verify thresholds.json is generated correctly
  - Test end-to-end prediction: text → labels → priority
  - Ask the user if questions arise

- [ ] 17. Implement GitHub Fetcher
  - [ ] 17.1 Create `src/utils/github_fetcher.py` with GitHubFetcher class
    - Implement `parse_url()` to extract owner, repo, issue_number from GitHub URL using regex
    - Implement `fetch_issue()` to call GitHub API: GET /repos/{owner}/{repo}/issues/{issue_number}
    - Implement `fetch_from_url()` as convenience method
    - Handle errors: invalid URL format, 404 not found, 403 rate limit, network errors
    - Return dict with title, body, and error information
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 17.2 Write property test for GitHub URL parsing
    - **Property 15: GitHub URL Parsing**
    - **Validates: Requirements 11.1**
    - Generate random valid GitHub URLs, verify correct extraction of owner/repo/number
    - _Requirements: 11.1_

  - [ ]* 17.3 Write property test for issue content concatenation
    - **Property 16: Issue Content Concatenation**
    - **Validates: Requirements 11.4**
    - Generate random title and body, verify concatenation format
    - _Requirements: 11.4_

  - [ ]* 17.4 Write unit tests for GitHub Fetcher
    - Test URL parsing with valid URLs
    - Test error handling for invalid URL format
    - Test error handling for API failures (use mocked requests)
    - Test rate limiting error message
    - Test 404 error message
    - _Requirements: 11.1, 11.3, 11.5_

- [ ] 18. Implement Model Explainer
  - [ ] 18.1 Create `src/utils/model_explainer.py` with ModelExplainer class
    - Initialize with model and preprocessor
    - Implement `explain_prediction()` using SHAP:
      - For Baseline: use SHAP LinearExplainer
      - For DistilBERT: use SHAP PartitionExplainer or sampling approach
    - Implement `visualize_explanation()` to create Plotly visualization of SHAP values
    - Highlight top 10 most influential tokens per label
    - _Requirements: 12.2, 12.3_

  - [ ]* 18.2 Write unit tests for Model Explainer
    - Test SHAP explanation generation with mocked model
    - Test visualization creation returns valid Plotly figure
    - Test error handling when SHAP fails
    - _Requirements: 12.2, 12.3_

- [ ] 19. Implement Streamlit Application
  - [ ] 19.1 Create `streamlit_app/app.py` main entry point
    - Set up page configuration and navigation
    - Load all models and artifacts on startup (with caching)
    - Create sidebar with page navigation
    - _Requirements: 10.1_

  - [ ] 19.2 Create `streamlit_app/pages/1_Text_Paste.py` for text input mode
    - Create text area for issue title + body input
    - Add submit button
    - On submit: preprocess → predict labels → predict priority
    - Display predicted labels with confidence scores
    - Display priority level with probabilities
    - Display needs_review flag status
    - Show SHAP explanation highlighting influential words
    - _Requirements: 10.2, 10.5, 10.7, 10.8_

  - [ ] 19.3 Create `streamlit_app/pages/2_GitHub_URL.py` for URL fetch mode
    - Create text input for GitHub issue URL
    - Add fetch button
    - On fetch: parse URL → fetch from GitHub API → display issue content
    - Run prediction pipeline on fetched content
    - Display same predictions as text paste mode
    - Handle and display errors (invalid URL, API failures, rate limiting)
    - _Requirements: 10.3, 10.6, 10.7, 10.8_

  - [ ] 19.4 Create `streamlit_app/pages/3_Model_Explorer.py` for model analysis
    - Create model comparison table (Baseline vs DistilBERT metrics)
    - Display per-label performance with Plotly bar charts
    - Visualize optimal thresholds per label
    - Show priority scorer feature importance using SHAP summary plot
    - Display confusion matrix for priority predictions
    - _Requirements: 10.4, 12.1, 12.4_

  - [ ] 19.5 Create `streamlit_app/utils/model_loader.py` for model loading
    - Implement functions to load all models and artifacts with caching
    - Load Dataset Manager artifacts (splits.json, label_mapping.json)
    - Load trained models (baseline, distilbert, priority scorer)
    - Load thresholds.json
    - Handle missing artifact errors gracefully
    - _Requirements: 13.3_

  - [ ]* 19.6 Write integration tests for Streamlit app
    - Test text paste mode end-to-end (use Streamlit testing framework)
    - Test GitHub URL mode with mocked API
    - Test model explorer page rendering
    - Test error display for various error conditions
    - _Requirements: 10.5, 10.6_

- [ ] 20. Create Kaggle notebook for model comparison
  - [ ] 20.1 Create `notebooks/06_model_comparison.ipynb`
    - Load both baseline and DistilBERT models
    - Evaluate both on test set
    - Create side-by-side performance comparison tables
    - Visualize per-label F1 scores for both models
    - Analyze prediction differences and error cases
    - Show example predictions with explanations
    - Document final performance metrics
    - _Requirements: 9.6_

- [ ] 21. Checkpoint - Verify application pipeline
  - Ensure all Streamlit pages render correctly
  - Test text paste mode with sample issues
  - Test GitHub URL mode with real GitHub issues
  - Verify model explorer displays all visualizations
  - Ask the user if questions arise

- [ ] 22. Prepare deployment artifacts
  - [ ] 22.1 Create deployment configuration files
    - Create `streamlit_app/requirements.txt` with production dependencies
    - Create `README.md` with setup and usage instructions
    - Create `.streamlit/config.toml` for Streamlit configuration
    - Ensure all artifacts are in correct locations for deployment
    - _Requirements: 13.3, 14.1, 14.2, 14.3, 14.4_

  - [ ] 22.2 Create Hugging Face Spaces deployment files
    - Create `app.py` in root directory (entry point for HF Spaces)
    - Create `requirements.txt` in root directory
    - Create `README.md` with model card information
    - Test deployment locally before pushing to HF Spaces
    - _Requirements: 13.1, 13.2, 13.3_

  - [ ]* 22.3 Write deployment verification tests
    - Test that all required artifacts are present
    - Test model loading time < 60 seconds
    - Test concurrent request handling (simulate 5 concurrent users)
    - _Requirements: 13.3, 13.5_

- [ ] 23. Final integration and testing
  - [ ] 23.1 Run complete test suite
    - Execute all unit tests (target: 80% coverage)
    - Execute all property tests (100 iterations each)
    - Execute integration tests
    - Generate coverage report
    - _Requirements: All_

  - [ ] 23.2 Verify all artifacts are committed
    - Commit splits.json to repository
    - Commit label_mapping.json to repository
    - Commit thresholds.json to repository
    - Commit or upload trained model weights
    - Verify all Kaggle notebooks are saved
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [ ] 23.3 Run end-to-end acceptance tests
    - Verify all 16 correctness properties pass
    - Verify performance targets met (Micro F1 ≥ 0.75, Macro F1 ≥ 0.65, Hamming Loss ≤ 0.15)
    - Verify inference latency < 3 seconds on CPU
    - Verify all 6 Kaggle notebooks execute successfully
    - Verify Streamlit app works end-to-end
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 24. Final checkpoint - Project completion
  - Ensure all tests pass
  - Verify all deliverables are complete (6 notebooks, Streamlit app, artifacts)
  - Verify deployment to Hugging Face Spaces is successful
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples, edge cases, and error conditions
- The implementation follows a bottom-up approach: data → preprocessing → models → application
- All models use Python with specified frameworks (TensorFlow/Keras, scikit-learn, XGBoost)
- Streamlit app provides three interaction modes: text paste, GitHub URL fetch, model explorer
- Final deployment target is Hugging Face Spaces with all artifacts included
