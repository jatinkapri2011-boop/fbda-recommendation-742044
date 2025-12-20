import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import faiss
import gdown
import os
import re

# =============================
# CONFIG
# =============================
GROUP_ID = "274244"
SEED = 274244
np.random.seed(SEED)

FILES = {
    "meta": ("meta_274244.parquet", "1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy"),
    "sample": ("sampled_10001_274244.csv", "1PBUEUc8N1XvaPnX8x9dbXy4iT_tNBSAl"),
    "tfidf": ("tfidf_matrix_274244.npz", "1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY"),
    "embed": ("embeddings_274244.npy", "1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq"),
    "faiss": ("faiss_index_274244.index", "17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu"),
}

# =============================
# DOWNLOAD FILES (CACHED)
# =============================
@st.cache_resource
def download_all_files():
    for fname, fid in FILES.values():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, fname, quiet=True)
    return True

download_all_files()

# =============================
# LOAD MODELS (CACHED)
# =============================
@st.cache_resource
def load_models():
    meta = pd.read_parquet(FILES["meta"][0])
    tfidf_matrix = sparse.load_npz(FILES["tfidf"][0])
    embeddings = np.load(FILES["embed"][0]).astype("float32")
    faiss_index = faiss.read_index(FILES["faiss"][0])
    return meta, tfidf_matrix, embeddings, faiss_index

meta, tfidf_matrix, embeddings, index = load_models()

movie_ids = meta["movie_id"].tolist()
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}


# =============================
# LOAD & PREPARE COLLAB (FAST, CACHED)
# =============================
@st.cache_resource
def load_collab():
    df = pd.read_csv(FILES["sample"][0])

    def parse_actors(s):
        return [a.strip() for a in re.split("[,|]", str(s)) if a.strip()]

    df["actors"] = df["stars"].apply(parse_actors)
  # Detect correct title column
title_col = None
for c in ["title", "Title", "movie_title", "name"]:
    if c in df.columns:
        title_col = c
        break

if title_col is None:
    raise KeyError("❌ Could not find a movie title column in CSV file")

# Build movie_id
df["movie_id"] = df[title_col].astype(str).str.strip() + " (" + df["year"].fillna("NA").astype(str) + ")"

    inter = df.explode("actors")[["actors", "movie_id"]]
    inter.columns = ["user", "item"]

    user_seen = inter.groupby("user")["item"].apply(set).to_dict()

    # Build an item popularity dictionary to sort recommendations
    popularity = df.groupby("movie_id")["rating"].mean().to_dict()

    return user_seen, popularity


user_seen, popularity = load_collab()


# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_tfidf(movie, n):
    i = movie_to_idx[movie]
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    sims[i] = -1
    top = sims.argsort()[::-1][:n]
    return [(movie_ids[j], float(sims[j])) for j in top]


def recommend_embed(movie, n):
    i = movie_to_idx[movie]
    D, I = index.search(embeddings[i:i+1], n + 1)
    out = []
    for d, j in zip(D[0], I[0]):
        if j != i:
            out.append((movie_ids[j], float(d)))
        if len(out) == n:
            break
    return out


def recommend_collab(actor, n):
    if actor not in user_seen:
        return []

    seen = user_seen[actor]
    candidates = [m for m in movie_ids if m not in seen]

    # Rank by movie popularity (FAST)
    ranked = sorted(
        [(m, popularity.get(m, 0)) for m in candidates],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:n]


# =============================
# UI
# =============================
st.set_page_config("FBDA Recommender", layout="wide")
st.title("🎬 Movie Recommendation System (Group 274244)")

st.sidebar.markdown("## 🔍 Select Recommendation Engine")

mode = st.sidebar.radio(
    "Recommendation Type",
    [
        "📄 Content-Based (TF-IDF)",
        "🧠 Semantic Embeddings",
        "👥 Collaborative Filtering (Actors)"
    ]
)

top_n = st.sidebar.slider("Top N Recommendations", 5, 20, 10)


# =============================
# MAIN LOGIC
# =============================
if mode.startswith("📄"):
    movie = st.selectbox("🎬 Choose a Movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)

elif mode.startswith("🧠"):
    movie = st.selectbox("🎬 Choose a Movie", movie_ids)
    recs = recommend_embed(movie, top_n)

else:
    actor = st.selectbox("🎭 Choose an Actor", sorted(user_seen.keys()))
    recs = recommend_collab(actor, top_n)


# =============================
# OUTPUT
# =============================
st.subheader("⭐ Recommendations")

df_out = pd.DataFrame(recs, columns=["Movie", "Score"])
st.dataframe(df_out, use_container_width=True)
