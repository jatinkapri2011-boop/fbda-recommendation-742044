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
GROUP_ID = "742044"
SEED = 742044
np.random.seed(SEED)

FILES = {
    "meta": ("meta_742044.parquet", "1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy"),
    "sample": ("sampled_10001_742044.csv", "1PBUEUc8N1XvaPnX8x9dbXy4iT_tNBSAl"),
    "tfidf": ("tfidf_matrix_742044.npz", "1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY"),
    "embed": ("embeddings_742044.npy", "1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq"),
    "faiss": ("faiss_index_742044.index", "17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu"),
}

# =============================
# DOWNLOAD FILES
# =============================
@st.cache_resource
def download_all_files():
    with st.status("📦 Preparing models (first run may take 1–2 minutes)...", expanded=True) as status:
        for key, (fname, fid) in FILES.items():
            if not os.path.exists(fname):
                st.write(f"⬇️ Downloading `{fname}`")
                url = f"https://drive.google.com/uc?id={fid}"
                gdown.download(url, fname, quiet=True)
            else:
                st.write(f"✅ `{fname}` already available")

        status.update(label="✅ All files ready", state="complete")

download_all_files()


# =============================
# LOAD DATA
# =============================
meta = pd.read_parquet("meta_742044.parquet")
tfidf_matrix = sparse.load_npz("tfidf_matrix_742044.npz")
embeddings = np.load("embeddings_742044.npy").astype("float32")
index = faiss.read_index("faiss_index_742044.index")

movie_ids = meta["movie_id"].tolist()
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

# =============================
# BUILD COLLABORATIVE DATA
# =============================
df = pd.read_csv("sampled_10001_742044.csv")

def parse_actors(s):
    return [a.strip() for a in re.split("[,|]", str(s)) if a.strip()]

df["actors"] = df["stars"].apply(parse_actors)
df["movie_id"] = df["title"].astype(str).str.strip() + " (" + df["year"].fillna("NA").astype(str) + ")"


interactions = df.explode("actors")[["actors", "movie_id", "rating"]]
interactions.columns = ["user", "item", "rating"]

users = interactions["user"].unique()
items = interactions["item"].unique()

user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {i: j for j, i in enumerate(items)}

# =============================
# MATRIX FACTORIZATION (NUMPY)
# =============================
K = 20
U = np.random.normal(scale=0.1, size=(len(users), K))
V = np.random.normal(scale=0.1, size=(len(items), K))

for _ in range(10):
    for _, row in interactions.iterrows():
        u = user_to_idx[row["user"]]
        i = item_to_idx[row["item"]]
        r = row["rating"]
        pred = U[u] @ V[i]
        err = r - pred
        U[u] += 0.01 * err * V[i]
        V[i] += 0.01 * err * U[u]

# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_collab(actor, n=10):
    if actor not in user_map:
        return []

    u = user_map[actor]
    scores = U[u] @ V.T
    top = np.argsort(scores)[::-1][:n]

    return [(items[i], float(scores[i])) for i in top]


def recommend_tfidf(movie, n=10):
    i = movie_to_idx[movie]
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    sims[i] = -1
    top = sims.argsort()[::-1][:n]
    return [(movie_ids[j], sims[j]) for j in top]

def recommend_embed(movie, n=10):
    i = movie_to_idx[movie]
    D, I = index.search(embeddings[i:i+1], n+1)
    return [(movie_ids[j], float(d)) for d, j in zip(D[0], I[0]) if j != i][:n]

# =============================
# STREAMLIT UI
# =============================
st.set_page_config("FBDA Recommender", layout="wide")
st.title("🎬 Movie Recommendation System (Group 742044)")
st.info(
    "⏳ First load may take 1–2 minutes due to model initialization. "
    "Subsequent usage will be fast thanks to caching."
)

mode = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["Content-Based", "Embedding-Based", "Collaborative Filtering"]
)

top_n = st.sidebar.slider("Top N", 5, 20, 10)

if mode == "Content-Based":
    movie = st.selectbox("Select Movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)

elif mode == "Embedding-Based":
    movie = st.selectbox("Select Movie", movie_ids)
    recs = recommend_embed(movie, top_n)

else:
    actor = st.selectbox("Select Actor", sorted(users))
    recs = recommend_collab(actor, top_n)

st.subheader("Recommendations")
st.dataframe(pd.DataFrame(recs, columns=["Movie", "Score"]))
