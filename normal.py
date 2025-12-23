import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
st.title(" PDF Topic Modeling")
st.write("Upload a PDF file to find topics")
uploaded_file = st.file_uploader("Choose PDF", type="pdf")
def extract_text(pdf_file):
    text = " "
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def make_chunks(text, size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks
if uploaded_file:
    raw_text = extract_text(uploaded_file)
    cleaned = clean_text(raw_text)
    st.text_area("Extracted Text", cleaned[:500] + "...", height=150)
    chunks = make_chunks(cleaned)
    st.write(f" Created {len(chunks)} chunks")
    if len(chunks) < 3:
        st.error("Text is too short! Upload a bigger PDF.")
        st.stop()
    st.write(" Finding topics...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=3)
    topics, probs = topic_model.fit_transform(chunks)
    st.success(" Topics found!")
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1]
    st.write(f"Total Topics: {len(topic_info)}**")
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = [word for word, score in topic_words[:5]]
            st.write(f" Topic {topic_id}")
            st.write(f"Keywords:** {', '.join(keywords)}")
            st.write(f"Chunks in this topic:** {count}")
            examples = [chunks[i] for i, t in enumerate(topics) if t == topic_id]
            if examples:
                st.text_area(f"Example", examples[0][:200] + "...", height=80, key=f"ex_{topic_id}")

            st.write("---")
        st.write(" Topic Hierarchy")
        try:
            hierarchical = topic_model.hierarchical_topics(chunks)
            fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical)
            st.plotly_chart(fig)
        except:
            st.info("Not enough data for hierarchy visualization")
    else:
        st.info(" Upload a PDF file to start")

