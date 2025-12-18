import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import faiss
import gdown
import os

# =============================
# GOOGLE DRIVE DOWNLOAD UTILITY
# =============================
def download(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False, fuzzy=True)

with st.spinner("Downloading model files (first run only)..."):
    download("1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy", "meta_742044.parquet")
    download("1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY", "tfidf_matrix_742044.npz")
    download("1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq", "embeddings_742044.npy")
    download("17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu", "faiss_index_742044.index")

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="FBDA Movie Recommendation System",
    layout="wide"
)

st.title("🎬 FBDA Movie Recommendation System (Group 742044)")

# =============================
# LOAD DATA & MODELS
# =============================
@st.cache_data
def load_meta():
    return pd.read_parquet("meta_742044.parquet")

@st.cache_resource
def load_models():
    tfidf_matrix = sparse.load_npz("tfidf_matrix_742044.npz")
    embeddings = np.load("embeddings_742044.npy").astype("float32")
    index = faiss.read_index("faiss_index_742044.index")
    return tfidf_matrix, embeddings, index

try:
    meta = load_meta()
    tfidf_matrix, embeddings, index = load_models()
    st.success("Models loaded successfully ✅")
except Exception as e:
    st.error("Failed to load data or models ❌")
    st.exception(e)
    st.stop()

# =============================
# PREPARATION
# =============================
movie_ids = meta["movie_id"].tolist()
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_tfidf(movie_id, top_n):
    idx = movie_to_idx[movie_id]
    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    scores[idx] = -1
    top_idx = scores.argsort()[::-1][:top_n]
    return [(movie_ids[i], scores[i]) for i in top_idx]

def recommend_embeddings(movie_id, top_n):
    idx = movie_to_idx[movie_id]
    query = embeddings[idx:idx+1]

    scores, indices = index.search(query, top_n + 1)

    results = []
    for score, i in zip(scores[0], indices[0]):
        if i != idx:
            results.append((movie_ids[i], float(score)))

    return results[:top_n]

# =============================
# UI
# =============================
method = st.sidebar.selectbox(
    "Select Recommendation Method",
    ["Content-Based (TF-IDF)", "Text Embeddings (FAISS)"]
)

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

selected_movie = st.selectbox("Select a Movie", movie_ids)

if method == "Content-Based (TF-IDF)":
    recs = recommend_tfidf(selected_movie, top_n)
else:
    recs = recommend_embeddings(selected_movie, top_n)

# =============================
# DISPLAY RESULTS
# =============================
st.subheader("🎯 Recommended Movies")

rows = []
for movie, score in recs:
    row = meta[meta["movie_id"] == movie].iloc[0].to_dict()
    row["score"] = round(float(score), 4)
    rows.append(row)

df = pd.DataFrame(rows)[
    ["movie_id", "year", "genre", "rating", "votes", "score"]
]

st.dataframe(df, use_container_width=True)

with st.expander("📄 Movie Descriptions"):
    for r in rows:
        st.markdown(f"**{r['movie_id']}**")
        st.write(r["description"])
        st.markdown("---")
