import streamlit as st
import pdfplumber
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
st.title("PDF Text Classifier with BERT")
st.write("Upload a PDF file")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-z0-9\s]', '', text)  
    return text.strip()
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF"):
        raw_text = extract_text_from_pdf(uploaded_file)
    st.subheader(" Extracted Text")
    st.text_area("Raw Text", raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text, height=150)
    cleaned_text = clean_text(raw_text)
    st.text_area("Cleaned Text", cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text, height=150)
    with st.spinner("Loading BERT model..."):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2  
        )
        model.eval()
    chunks = chunk_text(cleaned_text, chunk_size=300)
    st.write(f"Processing {len(chunks)} text chunk(s).")
    predictions = []
    probabilities = []
    with st.spinner("Classifying text..."):
        for chunk in chunks:
            tokens = tokenizer(
                chunk,
                padding='max_length',  
                truncation=True,       
                max_length=512,        
                return_tensors='pt'  
            )
            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1).item()
                predictions.append(pred)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probabilities.append(probs[0].tolist())
    final_prediction = max(set(predictions), key=predictions.count)
    st.write(f"Final Prediction: Class {final_prediction}")
    avg_prob_class_0  = sum([p[0] for p in probabilities]) / len(probabilities)
    avg_prob_class_1 = sum([p[1] for p in probabilities]) / len(probabilities)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Prediction", f"Class {final_prediction}")
        st.write("**Class Labels:**")
        st.write("- Class 0: Negative/Category A")
        st.write("- Class 1: Positive/Category B")
    
    with col2:
        st.metric("Confidence (Class 0)", f"{avg_prob_class_0:.2%}")
        st.metric("Confidence (Class 1)", f"{avg_prob_class_1:.2%}")
    with st.expander("View detailed predictions for each chunk"):
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            st.write(f"**Chunk {i+1}:** Class {pred} (Confidence: {prob[pred]:.2%})")
    
else:
    st.info("👆 Please upload a PDF file to get started")
    
