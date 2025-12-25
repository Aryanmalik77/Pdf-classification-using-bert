import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import re



st.title(" Topic Analysis")
st.write("Upload your book PDF to discover topics and their relationships")


with st.sidebar:
    st.header("Settings")
    words_per_chunk = st.slider("Words per chunk", 100, 300, 200)
    min_chunks_per_topic = st.slider("Min chunks per topic", 2, 10, 3)
    use_llm_enrichment = st.checkbox("🤖 Use LLM for chunk enrichment", value=False)

    if use_llm_enrichment:
        st.info(" LangChain + Hugging Face LLM")
        llm_model = st.selectbox(
            "Select LLM Model",
            [
                "google/flan-t5-small",
                "google/flan-t5-base",
                "facebook/bart-large-cnn",
            ],
            help="Smaller models are faster"
        )

        st.write("**Enrichment Strategy:**")
        enrichment_mode = st.radio(
            "Mode",
            ["Extract Concepts", "Summarize", "Identify Themes"],
            help="Different prompts for different analysis"
        )


uploaded_file = st.file_uploader("Upload Book PDF", type="pdf")

@st.cache_resource
def load_langchain_llm(model_name):
    try:
        if "bart" in model_name.lower():
            hf_pipeline = pipeline(
                "summarization",
                model=model_name,
                device=-1,
                max_length=150,
                min_length=30
            )
        else:
            hf_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=-1,
                max_length=150
            )

        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm

    except Exception as e:
        st.error(f"Failed to load LangChain LLM: {e}")
        return None


def create_langchain_prompts(mode):


    if mode == "Extract Concepts":
        template = """Extract the main concepts and key ideas from this text in 2-3 sentences:

Text: {text}

Main concepts:"""

    elif mode == "Summarize":
        template = """Provide a concise summary of this text focusing on the core message:

Text: {text}

Summary:"""

    else:
        template = """Identify the main themes and topics discussed in this text:

Text: {text}

Themes:"""

    return PromptTemplate(template=template, input_variables=["text"])
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


def enrich_chunks_with_langchain(chunks, llm, prompt_template, model_name):
    enriched_chunks = []
    chain = prompt_template | llm | StrOutputParser()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks):
        try:
            status_text.text(f" Processing chunk {i + 1}/{len(chunks)} with LangChain...")
            progress_bar.progress((i + 1) / len(chunks))
            max_input_length = 512 if "t5" in model_name.lower() else 1024
            words = chunk.split()
            if len(words) > max_input_length:
                chunk = " ".join(words[:max_input_length])
            result = chain.invoke({"text":chunk})
            key_concepts = result.strip()
            enriched_chunk = f"{chunk} [LLM_ENRICHMENT: {key_concepts}]"
            enriched_chunks.append(enriched_chunk)

        except Exception as e:
            st.warning(f" LangChain failed for chunk {i + 1}: {str(e)[:50]}...")
            enriched_chunks.append(chunk)

    progress_bar.empty()
    status_text.empty()
    return enriched_chunks
def find_topics(chunks, min_size=3):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_size,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(chunks)
    return topic_model, topics, probs


def create_hierarchical_tree_graph(hierarchical_topics, topic_model, topic_ids):
    import plotly.graph_objects as go
    from collections import defaultdict
    tree = defaultdict(list)
    root_topics = []

    for _, row in hierarchical_topics.iterrows():
        parent = row['Parent_ID']
        child = row['Topics']
        if parent != child:
            tree[parent].append(child)
        else:
            root_topics.append(child)
    if not tree:
        root_topics = topic_ids
    def assign_positions(node, level=0, pos_x=0.5, width=1.0, positions=None, levels=None):
        if positions is None:
            positions = {}
            levels = defaultdict(list)

        positions[node] = (pos_x, -level)
        levels[level].append(node)

        children = tree.get(node, [])
        if children:
            child_width = width / len(children)
            start_x = pos_x - width / 2 + child_width / 2

            for i, child in enumerate(children):
                child_x = start_x + i * child_width
                assign_positions(child, level + 1, child_x, child_width, positions, levels)

        return positions, levels
    all_positions = {}
    all_levels = defaultdict(list)

    if root_topics:
        root_width = 1.0 / len(root_topics)
        for i, root in enumerate(root_topics):
            root_x = i * root_width + root_width / 2
            positions, levels = assign_positions(root, 0, root_x, root_width)
            all_positions.update(positions)
            for level, nodes in levels.items():
                all_levels[level].extend(nodes)
    edge_traces = []
    for parent, children in tree.items():
        if parent in all_positions:
            parent_x, parent_y = all_positions[parent]
            for child in children:
                if child in all_positions:
                    child_x, child_y = all_positions[child]
                    edge_trace = go.Scatter(
                        x=[parent_x, child_x, None],
                        y=[parent_y, child_y, None],
                        mode='lines',
                        line=dict(width=2, color='#6c9bcf'),
                        hoverinfo='none',
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []

    for topic_id, (x, y) in all_positions.items():
        node_x.append(x)
        node_y.append(y)

        if topic_id in topic_ids:
            words = [w for w, s in topic_model.get_topic(topic_id)[:3]]
            node_text.append(f"T{topic_id}")
            node_hover.append(f"Topic {topic_id}<br>" + ", ".join(words))
            node_colors.append('#7ba3cc')
        else:
            node_text.append(f"T{topic_id}")
            node_hover.append(f"Merged Topic {topic_id}")
            node_colors.append('#a5c9e8')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(width=2, color='#4a7ba7')
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertext=node_hover,
        hoverinfo='text',
        showlegend=False
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Hierarchical Topic Organization Chart",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='rgba(240, 245, 250, 0.5)'
    )

    return fig


def get_topic_connections(topic_model, topic_ids, threshold=0.5):
    try:
        embeddings = []
        for tid in topic_ids:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                topic_embedding = topic_model.topic_embeddings_[topic_model._map_predictions([tid])[0]]
                embeddings.append(topic_embedding)

        if len(embeddings) == 0:
            return [], np.array([])

        embeddings = np.array(embeddings)
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

    except Exception as e:
        st.warning(f"Could not calculate topic connections: {e}")
        n = len(topic_ids)
        return [], np.eye(n)
if uploaded_file:
    st.write(" Step 1: Reading PDF...")
    text = read_pdf(uploaded_file)
    text = clean_text(text)

    if len(text) < 500:
        st.error(" Book is too short! Need more content.")
        st.stop()

    st.success(f"Read {len(text.split())} words from book")

    with st.expander(" Preview book content"):
        st.text(text[:500] + "...")
    st.write(" Step 2: Splitting book into sections...")
    chunks = split_into_chunks(text, chunk_size=words_per_chunk)
    st.success(f" Created {len(chunks)} sections")

    if len(chunks) < 5:
        st.warning(" Very few sections. Try a longer book or smaller chunk size.")
        st.stop()
    st.write(" Step 3: Finding topics in your book...")
    st.write(" Step 2: Splitting book into sections...")
    chunks = split_into_chunks(text, chunk_size=words_per_chunk)
    st.success(f" Created {len(chunks)} sections")

    if len(chunks) < 5:
        st.warning(" Very few sections. Try a longer book or smaller chunk size.")
        st.stop()
    if use_llm_enrichment:
        st.write(" Step 2.5: Enriching chunks with LangChain + Hugging Face...")

        with st.spinner(f" Loading {llm_model} into LangChain..."):
            llm = load_langchain_llm(llm_model)

        if llm:
            st.info(f"🔗 Using LangChain with {enrichment_mode} strategy")

            prompt_template = create_langchain_prompts(enrichment_mode)

            with st.expander(" View Prompt Template"):
                st.code(prompt_template.template, language="text")
            enriched_chunks = enrich_chunks_with_langchain(
                chunks,
                llm,
                prompt_template,
                llm_model
            )
            chunks = enriched_chunks
            st.success(" Chunks enriched with LangChain LLM!")
            with st.expander("🔍 View Example Enrichment"):
                if len(enriched_chunks) > 0:
                    example = enriched_chunks[0]
                    if "[LLM_ENRICHMENT:" in example:
                        original, enriched = example.split("[LLM_ENRICHMENT:")
                        st.write("**Original Chunk:**")
                        st.text(original[:200] + "...")
                        st.write("**LLM Enrichment:**")
                        st.info(enriched.replace("]", ""))
        else:
            st.error(" Failed to load LangChain LLM. Using original chunks.")
    st.write("🔍 Step 3: Finding topics in your book...")
    with st.spinner(" Analyzing with BERT..."):
        topic_model, topics, probs = find_topics(chunks, min_size=min_chunks_per_topic)
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]

    if len(valid_topics) == 0:
        st.error("No topics found! Try lowering 'Min chunks per topic'.")
        st.stop()

    topic_ids = valid_topics['Topic'].tolist()
    st.success(f" Found {len(topic_ids)} main topics")

    st.write("")
    st.write(" Topics Discovered in Your Book")

    for topic_id in topic_ids:
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = [word for word, score in topic_words[:8]]
            count = sum(1 for t in topics if t == topic_id)
            with st.expander(f"Topic {topic_id}: {', '.join(keywords[:5])} ({count} sections)"):
                st.write("**Keywords & Relevance:**")
                for word, score in topic_words[:8]:
                    st.write(f"• {word} - {score:.3f}")
                st.write(f"**Semantic Meaning:** This topic discusses {', '.join(keywords[:5])}")

                example_chunks = [chunks[i] for i, t in enumerate(topics) if t == topic_id]
                if example_chunks:
                    st.write("**Example section:**")
                    example_text = example_chunks[0].split('[LLM_ENRICHMENT:')[0]
                    st.text_area("", example_text[:250] + "...", height=100, key=f"topic_{topic_id}")

    st.write("---")
    st.write("## 🔗 Topic Connectivity")

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


    st.header("Hierarchical Topic Structure")

    try:
        with st.spinner("Building topic hierarchy..."):
            hierarchical_topics = topic_model.hierarchical_topics(chunks)



        # Traditional Dendrogram
        with st.expander(" View Traditional Dendrogram"):
            fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            st.plotly_chart(fig_hierarchy, use_container_width=True)

        st.success(
            " Topic hierarchy created! The organization chart shows how topics are related in a parent-child structure.")

        # Show hierarchy table
        with st.expander("View Hierarchy Details"):
            hierarchy_data = []
            for _, row in hierarchical_topics.iterrows():
                if row['Parent_ID'] != row['Topics']:
                    parent_words = [w for w, s in topic_model.get_topic(row['Parent_ID'])[:3]] if row[
                                                                                                      'Parent_ID'] in topic_ids else [
                        'Root']
                    child_words = [w for w, s in topic_model.get_topic(row['Topics'])[:3]] if row[
                                                                                                  'Topics'] in topic_ids else [
                        'Merged']

                    hierarchy_data.append({
                        'Parent Topic': f"T{row['Parent_ID']}: {', '.join(parent_words)}",
                        'Child Topic': f"T{row['Topics']}: {', '.join(child_words)}",
                        'Distance': f"{row['Distance']:.3f}"
                    })

            if hierarchy_data:
                hierarchy_df = pd.DataFrame(hierarchy_data)
                st.dataframe(hierarchy_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f" Could not create hierarchy: {e}")
        st.info("This might happen with very few topics. Try analyzing a")
    st.write("---")



else:


    st.info(" Upload a book PDF to start analysis")
