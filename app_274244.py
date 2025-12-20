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

# Google Drive file mappings
FILES = {
    "meta": ("meta_274244.parquet", "1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy"),
    "sample": ("sampled_10001_274244.csv", "1PBUEUc8N1XvaPnX8x9dbXy4iT_tNBSAl"),
    "tfidf": ("tfidf_matrix_274244.npz", "1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY"),
    "embed": ("embeddings_274244.npy", "1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq"),
    "faiss": ("faiss_index_274244.index", "17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu"),
    "tfidf_vec": ("tfidf_vectorizer_274244.pkl", "1h1brxLMYv8u4yH-_U70Kxwprv_u0hjEy"),
    "svd": ("svd_model_274244.pkl", "1yXZQPeLB9dao-9bsnM1SRUNnRc2sWfL7"),
}

# =============================
# DOWNLOAD FILES (CACHED)
# =============================
@st.cache_resource
def download_all_files():
    with st.status("📦 Preparing models… first run may take 1 minute...", expanded=True) as status:
        for key, (fname, fid) in FILES.items():
            if not os.path.exists(fname):
                st.write(f"⬇ Downloading `{fname}`")
                url = f"https://drive.google.com/uc?id={fid}"
                gdown.download(url, fname, quiet=True)
            else:
                st.write(f"✅ `{fname}` already exists")
        status.update(label="✅ All files downloaded", state="complete")

download_all_files()

# =============================
# LOAD DATA
# =============================
meta = pd.read_parquet("meta_274244.parquet")
tfidf_matrix = sparse.load_npz("tfidf_matrix_274244.npz")
embeddings = np.load("embeddings_274244.npy").astype("float32")
index = faiss.read_index("faiss_index_274244.index")

movie_ids = meta["movie_id"].tolist()
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

# =============================
# BUILD COLLABORATIVE DATA
# =============================
df = pd.read_csv("sampled_10001_274244.csv")

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
# MATRIX FACTORIZATION (FAST)
# =============================
K = 20
U = np.random.normal(scale=0.1, size=(len(users), K))
V = np.random.normal(scale=0.1, size=(len(items), K))

# Only 1 epoch → Fast enough for 10k records
for _, r in interactions.iterrows():
    u = user_to_idx[r["user"]]
    i = item_to_idx[r["item"]]
    err = r["rating"] - (U[u] @ V[i])
    U[u] += 0.01 * err * V[i]
    V[i] += 0.01 * err * U[u]

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
    actor = actor.strip()
    if actor not in user_to_idx:
        return []

    u = user_to_idx[actor]
    scores = U[u] @ V.T

    seen = user_seen.get(actor, set())
    ranked = np.argsort(scores)[::-1]

    output = []
    for idx in ranked:
        item = items[idx]
        if item not in seen:
            output.append((item, float(scores[idx])))
        if len(output) >= n:
            break
    return output

# =============================
# STREAMLIT UI
# =============================
st.set_page_config("FBDA Recommender 274244", layout="wide")

st.title("🎬 Movie Recommendation System — Group 274244")
st.info("⚡ Models loaded. Choose a recommendation method from the sidebar.")

st.sidebar.markdown("## ⚙ Recommendation Engine")

mode = st.sidebar.radio(
    "Select Recommendation Model",
    [
        "📄 Content-Based (TF-IDF)",
        "🧠 Semantic Embeddings",
        "👥 Collaborative Filtering (Actors)"
    ]
)

top_n = st.sidebar.slider("Top N Recommendations", 5, 25, 10)

# =============================
# USER SELECTION
# =============================
if mode == "📄 Content-Based (TF-IDF)":
    movie = st.selectbox("🎬 Select Movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)

elif mode == "🧠 Semantic Embeddings":
    movie = st.selectbox("🎬 Select Movie", movie_ids)
    recs = recommend_embed(movie, top_n)

else:
    actor = st.selectbox("🎭 Select Actor", sorted(users))
    recs = recommend_collab(actor, top_n)

# =============================
# OUTPUT
# =============================
st.subheader("✨ Recommendations")
df_out = pd.DataFrame(recs, columns=["Movie", "Score"])
st.dataframe(df_out, use_container_width=True)
