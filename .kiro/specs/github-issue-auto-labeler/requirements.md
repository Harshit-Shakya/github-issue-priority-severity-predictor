# Requirements Document

## Introduction

The GitHub Issue Auto-Labeler is an end-to-end NLP system that automatically assigns canonical labels to GitHub issues and predicts their priority levels. The system uses machine learning models to classify issues into categories (bug, feature, documentation, performance, question, good-first-issue, security) and assigns priority scores (LOW/MEDIUM/HIGH) based on deterministic rules and model predictions. This project serves as a 4th year B.Tech resume project with a 5-week part-time development timeline.

## Glossary

- **System**: The GitHub Issue Auto-Labeler application
- **Canonical_Labels**: The seven predefined labels (bug, feature, documentation, performance, question, good-first-issue, security)
- **Priority_Scorer**: The XGBoost model that predicts issue priority levels
- **Label_Classifier**: The machine learning model that assigns canonical labels to issues
- **Baseline_Model**: The TF-IDF + Logistic Regression model used for performance comparison
- **Main_Model**: The fine-tuned DistilBERT model for multi-label classification
- **Text_Preprocessor**: The component that cleans and normalizes issue text
- **Dataset_Manager**: The component that handles data loading, splitting, and storage
- **Threshold_Tuner**: The component that optimizes per-label classification thresholds
- **Streamlit_App**: The web application interface for the system
- **Needs_Review_Flag**: A boolean indicator when no label exceeds the confidence threshold
- **Pseudo_Labels**: Priority labels generated from deterministic rules based on issue labels

## Requirements

### Requirement 1: Data Acquisition and Management

**User Story:** As a developer, I want to load and manage the GitHub issues dataset, so that I can train and evaluate machine learning models consistently.

#### Acceptance Criteria

1. WHEN the system initializes, THE Dataset_Manager SHALL load the lewtun/github-issues dataset from HuggingFace
2. THE Dataset_Manager SHALL create an 80/10/10 train/validation/test split using a fixed random seed of 42
3. WHEN the data split is created, THE Dataset_Manager SHALL save the split indices to splits.json
4. THE Dataset_Manager SHALL create a canonical label mapping for the seven labels (bug, feature, documentation, performance, question, good-first-issue, security)
5. WHEN the label mapping is created, THE Dataset_Manager SHALL save it to label_mapping.json
6. THE Dataset_Manager SHALL persist splits.json and label_mapping.json to the repository for reproducibility

### Requirement 2: Text Preprocessing Pipeline

**User Story:** As a data scientist, I want to preprocess GitHub issue text consistently, so that models receive clean, normalized input.

#### Acceptance Criteria

1. WHEN raw issue text is provided, THE Text_Preprocessor SHALL remove markdown formatting syntax
2. WHEN issue text contains code blocks, THE Text_Preprocessor SHALL extract and normalize code block content
3. WHEN issue text contains stack traces, THE Text_Preprocessor SHALL normalize stack trace formatting
4. WHEN issue text contains URLs, THE Text_Preprocessor SHALL replace URLs with a standardized token
5. THE Text_Preprocessor SHALL preserve semantic meaning while normalizing text format
6. FOR ALL valid issue text inputs, preprocessing then formatting then preprocessing SHALL produce equivalent normalized text

### Requirement 3: Baseline Model Implementation

**User Story:** As a machine learning engineer, I want to implement a baseline model, so that I can establish performance benchmarks for comparison.

#### Acceptance Criteria

1. THE Baseline_Model SHALL use TF-IDF vectorization for feature extraction
2. THE Baseline_Model SHALL use Logistic Regression from scikit-learn for multi-label classification
3. WHEN the Baseline_Model is trained, THE System SHALL train it on the training split from splits.json
4. WHEN the Baseline_Model makes predictions, THE System SHALL return probability scores for all seven Canonical_Labels
5. THE Baseline_Model SHALL support multi-label classification where issues can have multiple labels

### Requirement 4: DistilBERT Model Fine-Tuning

**User Story:** As a machine learning engineer, I want to fine-tune a DistilBERT model, so that I can achieve high-quality multi-label classification.

#### Acceptance Criteria

1. THE Main_Model SHALL use DistilBERT as the base architecture
2. THE Main_Model SHALL be implemented using TensorFlow/Keras
3. WHEN the Main_Model is trained, THE System SHALL fine-tune it on the training split from splits.json
4. WHEN the Main_Model makes predictions, THE System SHALL return probability scores for all seven Canonical_Labels
5. THE Main_Model SHALL support multi-label classification where issues can have multiple labels
6. WHEN training completes, THE System SHALL save the fine-tuned model weights for inference

### Requirement 5: Performance Metrics and Validation

**User Story:** As a machine learning engineer, I want to measure model performance against defined targets, so that I can validate the system meets quality requirements.

#### Acceptance Criteria

1. WHEN evaluating on the test set, THE System SHALL calculate Micro F1 score and it SHALL be ≥ 0.75
2. WHEN evaluating on the test set, THE System SHALL calculate Macro F1 score and it SHALL be ≥ 0.65
3. WHEN evaluating on the test set, THE System SHALL calculate per-label F1 scores and each SHALL be ≥ 0.50
4. WHEN evaluating the security label on the test set, THE System SHALL achieve F1 score ≥ 0.35
5. WHEN evaluating on the test set, THE System SHALL calculate Hamming Loss and it SHALL be ≤ 0.15
6. WHEN making predictions, THE System SHALL complete inference within 3 seconds on CPU

### Requirement 6: Threshold Optimization

**User Story:** As a machine learning engineer, I want to optimize classification thresholds per label, so that I can maximize prediction accuracy.

#### Acceptance Criteria

1. THE Threshold_Tuner SHALL optimize classification thresholds independently for each of the seven Canonical_Labels
2. WHEN optimizing thresholds, THE Threshold_Tuner SHALL use the validation split from splits.json
3. WHEN threshold optimization completes, THE System SHALL save the optimized thresholds to thresholds.json
4. THE System SHALL persist thresholds.json to the repository for reproducibility
5. WHEN making predictions, THE Label_Classifier SHALL apply the per-label thresholds from thresholds.json

### Requirement 7: Needs Review Flag

**User Story:** As a user, I want to know when the model is uncertain about label assignments, so that I can manually review ambiguous cases.

#### Acceptance Criteria

1. WHEN the Label_Classifier makes predictions, THE System SHALL compare each label probability against its threshold
2. IF no label probability exceeds its corresponding threshold, THEN THE System SHALL set the Needs_Review_Flag to true
3. IF at least one label probability exceeds its corresponding threshold, THEN THE System SHALL set the Needs_Review_Flag to false
4. WHEN returning predictions, THE System SHALL include the Needs_Review_Flag in the response

### Requirement 8: Priority Scoring System

**User Story:** As a project manager, I want to automatically assign priority levels to issues, so that I can triage work effectively.

#### Acceptance Criteria

1. THE Priority_Scorer SHALL use XGBoost as the classification algorithm
2. THE Priority_Scorer SHALL predict one of three priority levels: LOW, MEDIUM, or HIGH
3. WHEN generating training data, THE System SHALL create Pseudo_Labels using deterministic rules
4. WHEN an issue has the security label, THE System SHALL assign HIGH priority in Pseudo_Labels
5. WHEN an issue has both bug and performance labels, THE System SHALL assign HIGH priority in Pseudo_Labels
6. WHEN an issue has the good-first-issue label, THE System SHALL assign LOW priority in Pseudo_Labels
7. WHEN an issue has the question label, THE System SHALL assign LOW priority in Pseudo_Labels
8. THE Priority_Scorer SHALL use the predicted Canonical_Labels as input features

### Requirement 9: Kaggle Notebook Deliverables

**User Story:** As a developer, I want to document the development process in Kaggle notebooks, so that the work is reproducible and shareable.

#### Acceptance Criteria

1. THE System SHALL include a Kaggle notebook for exploratory data analysis (EDA)
2. THE System SHALL include a Kaggle notebook for data preprocessing implementation
3. THE System SHALL include a Kaggle notebook for baseline model training and evaluation
4. THE System SHALL include a Kaggle notebook for DistilBERT fine-tuning
5. THE System SHALL include a Kaggle notebook for priority scorer training
6. THE System SHALL include a Kaggle notebook for model comparison and analysis
7. WHEN notebooks are executed in sequence, THE System SHALL produce all required artifacts (splits.json, label_mapping.json, thresholds.json, model weights)

### Requirement 10: Streamlit Application Interface

**User Story:** As a user, I want to interact with the system through a web interface, so that I can easily classify GitHub issues.

#### Acceptance Criteria

1. THE Streamlit_App SHALL provide three distinct pages for different interaction modes
2. THE Streamlit_App SHALL include a text paste mode page where users can input issue text directly
3. THE Streamlit_App SHALL include a GitHub URL fetch mode page where users can provide issue URLs
4. THE Streamlit_App SHALL include a model explorer page for analyzing model behavior
5. WHEN a user submits issue text in text paste mode, THE Streamlit_App SHALL display predicted labels and priority
6. WHEN a user submits a GitHub URL in fetch mode, THE Streamlit_App SHALL retrieve the issue content and display predictions
7. WHEN predictions are displayed, THE Streamlit_App SHALL show the Needs_Review_Flag status
8. WHEN predictions are displayed, THE Streamlit_App SHALL show confidence scores for each label

### Requirement 11: GitHub URL Fetching

**User Story:** As a user, I want to fetch issue content from GitHub URLs, so that I can classify existing issues without manual copying.

#### Acceptance Criteria

1. WHEN a user provides a GitHub issue URL, THE System SHALL parse the URL to extract repository and issue number
2. WHEN the URL is valid, THE System SHALL fetch the issue title and body from GitHub
3. IF the GitHub API request fails, THEN THE System SHALL return a descriptive error message
4. WHEN issue content is fetched, THE System SHALL combine title and body for classification
5. THE System SHALL handle GitHub API rate limiting gracefully

### Requirement 12: Model Explainability

**User Story:** As a user, I want to understand why the model made specific predictions, so that I can trust and validate the results.

#### Acceptance Criteria

1. WHERE the model explorer page is accessed, THE Streamlit_App SHALL provide visualization of model predictions
2. THE Streamlit_App SHALL use SHAP values to explain individual predictions
3. WHEN displaying explanations, THE Streamlit_App SHALL highlight text features that influenced each label prediction
4. THE Streamlit_App SHALL display feature importance scores for the Priority_Scorer

### Requirement 13: Deployment to Hugging Face Spaces

**User Story:** As a developer, I want to deploy the application to Hugging Face Spaces, so that it is publicly accessible.

#### Acceptance Criteria

1. THE System SHALL be deployable to Hugging Face Spaces
2. WHEN deployed, THE Streamlit_App SHALL be accessible via a public URL
3. THE deployment SHALL include all required model artifacts (model weights, thresholds.json, label_mapping.json)
4. THE deployment SHALL handle concurrent user requests without degradation
5. WHEN the deployment starts, THE System SHALL load all models within 60 seconds

### Requirement 14: Artifact Version Control

**User Story:** As a developer, I want to version control all generated artifacts, so that experiments are reproducible.

#### Acceptance Criteria

1. THE System SHALL commit splits.json to the repository
2. THE System SHALL commit label_mapping.json to the repository
3. THE System SHALL commit thresholds.json to the repository
4. THE System SHALL commit trained model weights to the repository or model registry
5. WHEN artifacts are updated, THE System SHALL maintain backward compatibility with existing code

### Requirement 15: Technology Stack Compliance

**User Story:** As a developer, I want to use the specified technology stack, so that the project meets technical requirements.

#### Acceptance Criteria

1. THE Main_Model SHALL be implemented using TensorFlow/Keras, not PyTorch
2. THE System SHALL use HuggingFace datasets library for data loading
3. THE Baseline_Model SHALL use scikit-learn for implementation
4. THE Priority_Scorer SHALL use XGBoost for implementation
5. THE System SHALL use SHAP for model explainability
6. THE System SHALL use TextBlob for any text analysis features
7. THE Streamlit_App SHALL use Streamlit framework for the web interface
8. THE Streamlit_App SHALL use Plotly for interactive visualizations
