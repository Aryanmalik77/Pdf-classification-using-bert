import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

st.title(" Topic Analysis")
st.write("Upload your book PDF to discover topics and their relationships")

# Simple settings in sidebar
with st.sidebar:
    st.header("Settings")
    words_per_chunk = st.slider("Words per chunk", 100, 300, 200)
    min_chunks_per_topic = st.slider("Min chunks per topic", 2, 10, 3)

# File upload
uploaded_file = st.file_uploader("Upload Book PDF", type="pdf")


def read_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Only keep meaningful chunks
            chunks.append(chunk)
    return chunks


def find_topics(chunks, min_size=3):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_size,
        verbose=False
    )


    topics, probs = topic_model.fit_transform(chunks)

    return topic_model, topics, probs


def get_topic_connections(topic_model, topic_ids, threshold=0.5):
    embeddings = topic_model.topic_embeddings_[[tid + 1 for tid in topic_ids]]
    similarity = cosine_similarity(embeddings)
    connections = []
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            if similarity[i][j] > threshold:
                connections.append({
                    'Topic A': topic_ids[i],
                    'Topic B': topic_ids[j],
                    'Connection Strength': similarity[i][j]
                })

    return connections, similarity
if uploaded_file:

    # Step 1: Read PDF
    st.write(" Step 1: Reading PDF...")
    text = read_pdf(uploaded_file)
    text = clean_text(text)

    if len(text) < 500:
        st.error(" Book is too short! Need more content.")
        st.stop()

    st.success(f"Read {len(text.split())} words from book")

    # Show sample text
    with st.expander(" Preview book content"):
        st.text(text[:500] + "...")

    # Step 2: Split into chunks
    st.write(" Step 2: Splitting book into sections...")
    chunks = split_into_chunks(text, chunk_size=words_per_chunk)
    st.success(f" Created {len(chunks)} sections")

    if len(chunks) < 5:
        st.warning(" Very few sections. Try a longer book or smaller chunk size.")
        st.stop()
    st.write(" Step 3: Finding topics in your book...")
    with st.spinner(" Analyzing with BERT... (this takes a minute)"):
        topic_model, topics, probs = find_topics(chunks, min_size=min_chunks_per_topic)
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]

    if len(valid_topics) == 0:
        st.error(" No topics found! Try lowering 'Min chunks per topic'.")
        st.stop()

    topic_ids = valid_topics['Topic'].tolist()
    st.success(f" Found {len(topic_ids)} main topics")
    st.write("---")
    st.write("  Topics Discovered in Your Book")
    for topic_id in topic_ids:
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = [word for word, score in topic_words[:8]]
            count = sum(1 for t in topics if t == topic_id)
            with st.expander(f"Topic {topic_id}: {', '.join(keywords[:5])} ({count} sections)"):
                st.write("Keywords & Relevance:")
                for word, score in topic_words[:8]:
                    st.write(f"• {word} - {score:.3f}")
                st.write(f"Semantic Meaning:** This topic discusses {', '.join(keywords[:5])}**")
                example_chunks = [chunks[i] for i, t in enumerate(topics) if t == topic_id]
                if example_chunks:
                    st.write("Example section:")
                    st.text_area("", example_chunks[0][:250] + "...", height=100, key=f"topic_{topic_id}")


    st.write("---")
    st.write("##  Topic Connectivity (How Topics Relate)")

    with st.spinner("Analyzing connections..."):
        connections, similarity_matrix = get_topic_connections(topic_model, topic_ids, threshold=0.5)

    if connections:
        st.success(f" Found {len(connections)} connections between topics")
        conn_df = pd.DataFrame(connections)
        conn_data = []
        for _, row in conn_df.iterrows():
            topic_a_words = [w for w, s in topic_model.get_topic(row['Topic A'])[:3]]
            topic_b_words = [w for w, s in topic_model.get_topic(row['Topic B'])[:3]]

            conn_data.append({
                'From': f"Topic {row['Topic A']}: {', '.join(topic_a_words)}",
                'To': f"Topic {row['Topic B']}: {', '.join(topic_b_words)}",
                'Strength': f"{row['Connection Strength']:.2f}",
                'Type': '🔴 Strong' if row['Connection Strength'] > 0.7 else '🟡 Medium'
            })

        conn_display = pd.DataFrame(conn_data)
        st.dataframe(conn_display, use_container_width=True, hide_index=True)
        st.write("Connection Statistics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Connections", len(connections))
        with col2:
            strong = sum(1 for c in connections if c['Connection Strength'] > 0.7)
            st.metric("Strong Connections", strong)

    else:
        st.warning("No strong connections found. Topics are relatively independent.")

    st.write("---")
    st.write(" Similarity Matrix (How Similar Each Topic Is)")
    similarity_data = []
    for i, topic_i in enumerate(topic_ids):
        row_data = {'Topic': f"T{topic_i}"}
        for j, topic_j in enumerate(topic_ids):
            row_data[f"T{topic_j}"] = f"{similarity_matrix[i][j]:.2f}"
        similarity_data.append(row_data)

    sim_df = pd.DataFrame(similarity_data)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

    st.info("Values close to 1.00 = very similar topics. Values close to 0.00 = different topics.")


    st.write("---")
    st.write("##  Topic Hierarchy (Tree Structure)")

    try:
        hierarchical_topics = topic_model.hierarchical_topics(chunks)
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        st.plotly_chart(fig, use_container_width=True)
        st.success(" Hierarchy shows how topics group together like a family tree")
    except Exception as e:
        st.warning("Could not create hierarchy. Need more data or topics.")

else:

    st.info(" Upload a book PDF to start analysis")