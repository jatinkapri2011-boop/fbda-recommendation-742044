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
# FAST FILE DOWNLOAD
# =============================
@st.cache_resource
def download_files():
    for name, (fname, fid) in FILES.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, fname, quiet=True)
    return True

download_files()

# =============================
# LOAD DATA (CACHED)
# =============================
@st.cache_resource
def load_all_data():
    meta = pd.read_parquet("meta_274244.parquet")
    tfidf_matrix = sparse.load_npz("tfidf_matrix_274244.npz")
    embeddings = np.load("embeddings_274244.npy").astype("float32")
    index = faiss.read_index("faiss_index_274244.index")
    df = pd.read_csv("sampled_10001_274244.csv")
    return meta, tfidf_matrix, embeddings, index, df

meta, tfidf_matrix, embeddings, index, df = load_all_data()

movie_ids = meta["movie_id"].tolist()
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

# =============================
# COLLABORATIVE DATA CLEAN
# =============================
def parse_actors(s):
    return [a.strip() for a in re.split("[,|]", str(s)) if a.strip()]

df["actors"] = df["stars"].apply(parse_actors)
df["movie_id"] = df["Title"].astype(str).str.strip() + " (" + df["year"].fillna("NA").astype(str) + ")"

interactions = df.explode("actors")[["actors", "movie_id", "rating"]]
interactions.columns = ["user", "item", "rating"]

user_seen = interactions.groupby("user")["item"].apply(set).to_dict()

users = interactions["user"].unique()
items = interactions["item"].unique()

user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {i: j for j, i in enumerate(items)}

# =============================
# FAST MATRIX FACTORIZATION (CACHED)
# =============================
@st.cache_resource
def train_mf():
    K = 20
    U = np.random.normal(scale=0.1, size=(len(users), K))
    V = np.random.normal(scale=0.1, size=(len(items), K))

    # Only 1 fast pass (your original logic but now cached)
    for _, row in interactions.iterrows():
        u = user_to_idx[row["user"]]
        i = item_to_idx[row["item"]]
        r = row["rating"]
        err = r - (U[u] @ V[i])
        U[u] += 0.01 * err * V[i]
        V[i] += 0.01 * err * U[u]

    return U, V

U, V = train_mf()

# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_tfidf(movie, n=10):
    i = movie_to_idx[movie]
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    sims[i] = -1
    top = sims.argsort()[::-1][:n]
    return [(movie_ids[j], float(sims[j])) for j in top]

def recommend_embed(movie, n=10):
    i = movie_to_idx[movie]
    D, I = index.search(embeddings[i:i+1], n + 1)
    return [(movie_ids[j], float(d)) for d, j in zip(D[0], I[0]) if j != i][:n]

def recommend_collab(actor, n=10):
    if actor not in user_to_idx:
        return []
    u = user_to_idx[actor]
    scores = U[u] @ V.T
    seen = user_seen.get(actor, set())
    ranked = np.argsort(scores)[::-1]
    out = []
    for idx in ranked:
        item = items[idx]
        if item not in seen:
            out.append((item, float(scores[idx])))
        if len(out) >= n:
            break
    return out

# =============================
# STREAMLIT UI
# =============================
st.set_page_config("FBDA 274244", layout="wide")
st.title("🎬 Movie Recommendation System — Group 274244")

mode = st.sidebar.radio(
    "Select Recommendation Model",
    ["📄 Content-Based (TF-IDF)", "🧠 Semantic Embeddings", "👥 Collaborative Filtering (Actors)"]
)

top_n = st.sidebar.slider("Top N Recommendations", 5, 25, 10)

if mode == "📄 Content-Based (TF-IDF)":
    movie = st.selectbox("🎬 Select Movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)

elif mode == "🧠 Semantic Embeddings":
    movie = st.selectbox("🎬 Select Movie", movie_ids)
    recs = recommend_embed(movie, top_n)

else:
    actor = st.selectbox("🎭 Select Actor", sorted(users))
    recs = recommend_collab(actor, top_n)

st.subheader("Recommendations")
st.dataframe(pd.DataFrame(recs, columns=["Movie", "Score"]), use_container_width=True)
