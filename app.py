import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import re

# Model definition (same as training)
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, n_priority_classes=3, n_severity_classes=3, dropout=0.3):
        super(MultiTaskDistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        hidden_size = self.distilbert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        self.priority_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.priority_dropout = nn.Dropout(dropout)
        self.priority_classifier = nn.Linear(hidden_size // 2, n_priority_classes)
        
        self.severity_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.severity_dropout = nn.Dropout(dropout)
        self.severity_classifier = nn.Linear(hidden_size // 2, n_severity_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        priority_hidden = self.relu(self.priority_hidden(cls_output))
        priority_hidden = self.priority_dropout(priority_hidden)
        priority_logits = self.priority_classifier(priority_hidden)
        
        severity_hidden = self.relu(self.severity_hidden(cls_output))
        severity_hidden = self.severity_dropout(severity_hidden)
        severity_logits = self.severity_classifier(severity_hidden)
        
        return priority_logits, severity_logits

# Text preprocessing
def clean_text(text):
    if not text:
        return ''
    
    text = str(text)
    text = re.sub(r'```[\s\S]*?```', ' [CODE] ', text)
    text = re.sub(r'`[^`\n]+`', ' [CODE] ', text)
    text = re.sub(r'https?://\S+', ' [URL] ', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_title_body(title, body):
    clean_title = clean_text(title)
    clean_body = clean_text(body)
    
    if clean_title and clean_body:
        return f"[TITLE] {clean_title} [TITLE] {clean_title} [SEP] {clean_body}"
    elif clean_title:
        return f"[TITLE] {clean_title} [TITLE] {clean_title}"
    elif clean_body:
        return f"[SEP] {clean_body}"
    else:
        return ""

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskDistilBERT()
    
    try:
        model.load_state_dict(torch.load('models/distilbert_multitask_final.pth', map_location=device))
    except:
        try:
            model.load_state_dict(torch.load('distilbert_multitask_final.pth', map_location=device))
        except:
            st.error("Model file not found. Please ensure 'distilbert_multitask_final.pth' is in the current directory or 'models/' folder.")
            return None, None, None
    
    model.to(device)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    return model, tokenizer, device

# Example callbacks
def set_example(title_value, body_value):
    st.session_state.title = title_value
    st.session_state.body = body_value

# Prediction function
def predict(title, body, model, tokenizer, device):
    text = combine_title_body(title, body)
    
    encoding = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        priority_logits, severity_logits = model(input_ids, attention_mask)
    
    priority_pred = torch.argmax(priority_logits, dim=1).item()
    severity_pred = torch.argmax(severity_logits, dim=1).item()
    
    priority_probs = torch.softmax(priority_logits, dim=1)[0]
    severity_probs = torch.softmax(severity_logits, dim=1)[0]
    
    priority_labels = ['low', 'medium', 'high']
    severity_labels = ['Minor', 'Major', 'Critical']
    
    return {
        'priority': priority_labels[priority_pred],
        'priority_confidence': priority_probs[priority_pred].item(),
        'priority_probs': {label: prob.item() for label, prob in zip(priority_labels, priority_probs)},
        'severity': severity_labels[severity_pred],
        'severity_confidence': severity_probs[severity_pred].item(),
        'severity_probs': {label: prob.item() for label, prob in zip(severity_labels, severity_probs)}
    }

# Streamlit UI
st.title("GitHub Issue Priority & Severity Predictor")
st.write("Predict priority and severity for GitHub issues using DistilBERT multi-task model")

# Load model
model, tokenizer, device = load_model()

if model is not None:
    st.write("---")
    
    # Input fields with persistent state
    if 'title' not in st.session_state:
        st.session_state.title = ''
    if 'body' not in st.session_state:
        st.session_state.body = ''

    # Example issues
    with st.expander("Try Example Issues"):
        st.button(
            "Example 1: Critical Bug",
            key="example_1",
            on_click=set_example,
            args=(
                "Application crashes on startup",
                "The application crashes immediately after launching. Error message: 'Segmentation fault'. This affects all users on Linux.",
            ),
        )
        st.button(
            "Example 2: Feature Request",
            key="example_2",
            on_click=set_example,
            args=(
                "Add dark mode support",
                "It would be great to have a dark mode option for better usability at night.",
            ),
        )
        st.button(
            "Example 3: Documentation",
            key="example_3",
            on_click=set_example,
            args=(
                "Update installation instructions",
                "The installation guide is outdated. Please update it to reflect the new setup process.",
            ),
        )

    title = st.text_input("Issue Title", placeholder="e.g., Bug: Application crashes on startup", key='title')
    body = st.text_area("Issue Body", placeholder="Describe the issue in detail...", height=150, key='body')
    
    # Predict button
    if st.button("Predict", type="primary"):
        if not title and not body:
            st.warning("Please enter at least a title or body")
        else:
            with st.spinner("Analyzing issue..."):
                result = predict(title, body, model, tokenizer, device)
            
            st.write("---")
            st.subheader("Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Priority", result['priority'].upper(), 
                         f"{result['priority_confidence']*100:.1f}% confidence")
                
                st.write("**Priority Probabilities:**")
                for label, prob in result['priority_probs'].items():
                    st.progress(prob, text=f"{label}: {prob*100:.1f}%")
            
            with col2:
                st.metric("Severity", result['severity'].upper(), 
                         f"{result['severity_confidence']*100:.1f}% confidence")
                
                st.write("**Severity Probabilities:**")
                for label, prob in result['severity_probs'].items():
                    st.progress(prob, text=f"{label}: {prob*100:.1f}%")
    
    st.write("---")
    st.caption("Model: DistilBERT Multi-Task | Dataset: 114K GitHub Issues | Weighted F1: 0.80")
else:
    st.error("Failed to load model. Please check the model file path.")
