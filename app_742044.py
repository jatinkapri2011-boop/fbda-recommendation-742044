import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
import faiss
import gdown
import os

# =============================
# GOOGLE DRIVE DOWNLOAD HELPER
# =============================
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False, fuzzy=True)

with st.spinner("Downloading model files (first run only)..."):
    download_from_drive("1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy", "meta_742044.parquet")
    download_from_drive("1PBUEUc8N1XvaPnX8x9dbXy4iT_tNBSAl", "sampled_10001_742044.csv")
    download_from_drive("1yXZQPeLB9dao-9bsnM1SRUNnRc2sWfL7", "svd_model_742044.pkl")
    download_from_drive("1h1brxLMYv8u4yH-_U70Kxwprv_u0hjEy", "tfidf_vectorizer_742044.pkl")
    download_from_drive("1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY", "tfidf_matrix_742044.npz")
    download_from_drive("1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq", "embeddings_742044.npy")
    download_from_drive("17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu", "faiss_index_742044.index")

# =============================
# CONFIG
# =============================
GROUP_ID = "742044"
SEED = 742044

st.set_page_config(page_title=f"Movie Recommender {GROUP_ID}", layout="wide")
st.title(f"🎬 Movie Recommendation System (Group {GROUP_ID})")

# =============================
# LOAD DATA & MODELS
# =============================
@st.cache_data
def load_meta():
    return pd.read_parquet("meta_742044.parquet")

@st.cache_resource
def load_models()
:
    svd = joblib.load("svd_model_742044.pkl")
    tfidf = joblib.load("tfidf_vectorizer_742044.pkl")
    tfidf_matrix = sparse.load_npz("tfidf_matrix_742044.npz")
    emb = np.load("embeddings_742044.npy").astype("float32")
    index = faiss.read_index("faiss_index_742044.index")
    return svd, tfidf, tfidf_matrix, emb, index

try:
    meta = load_meta()
    svd, tfidf, tfidf_matrix, emb, index = load_models()
    st.success("Models and data loaded successfully ✅")
except Exception as e:
    st.error("Failed to load models or data")
    st.exception(e)
    st.stop()

# =============================
# PREP
# =============================
movie_ids = meta["movie_id"].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

# =============================
# ACTOR → MOVIES
# =============================
@st.cache_data
def build_user_seen():
    import re
    df = pd.read_csv("sampled_10001_742044.csv")

    def make_movie_id(row):
        y = row["year"]
        y = "NA" if pd.isna(y) else str(int(float(y)))
        return f'{row["Title"].strip()} ({y})'

    def parse_actors(x):
        return [p.strip() for p in re.split(r"[,\|]", str(x)) if p.strip()]

    df["movie_id"] = df.apply(make_movie_id, axis=1)
    df["actors"] = df["stars"].apply(parse_actors)

    inter = df[["movie_id", "rating", "actors"]].explode("actors")
    inter = inter.rename(columns={"actors": "user_id", "movie_id": "item_id"})
    user_seen = inter.groupby("user_id")["item_id"].apply(set).to_dict()

    return user_seen, sorted(user_seen.keys())

user_seen, actors = build_user_seen()

# =============================
# RECOMMENDERS
# =============================
def recommend_tfidf(movie, top_n):
    i = movie_to_idx[movie]
    sims = linear_kernel(tfidf_matrix[i], tfidf_matrix).flatten()
    sims[i] = -1
    idx = np.argsort(sims)[::-1][:top_n]
    return [(movie_ids[j], sims[j]) for j in idx]

def recommend_embed(movie, top_n):
    i = movie_to_idx[movie]
    q = emb[i:i+1]
    scores, idxs = index.search(q, top_n + 1)
    return [(movie_ids[j], float(s)) for s, j in zip(scores[0], idxs[0]) if j != i][:top_n]

def recommend_svd(actor, top_n):
    unseen = [m for m in movie_ids if m not in user_seen.get(actor, set())]
    scores = [(m, svd.predict(actor, m).est) for m in unseen[:2000]]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

# =============================
# UI
# =============================
mode = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "Content-Based (TF-IDF + Cosine)",
        "Text Embeddings (MiniLM + FAISS)",
        "Collaborative Filtering (Actors + SVD)"
    ]
)

top_n = st.sidebar.slider("Top N", 5, 20, 10)

if mode.startswith("Content"):
    movie = st.selectbox("Select a movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)
elif mode.startswith("Text"):
    movie = st.selectbox("Select a movie", movie_ids)
    recs = recommend_embed(movie, top_n)
else:
    actor = st.selectbox("Select an actor", actors)
    recs = recommend_svd(actor, top_n)

# =============================
# OUTPUT
# =============================
rows = []
for mid, score in recs:
    r = meta[meta["movie_id"] == mid].iloc[0].to_dict()
    r["score"] = score
    rows.append(r)

df_out = pd.DataFrame(rows)[["movie_id", "year", "genre", "rating", "votes", "score"]]
st.dataframe(df_out, use_container_width=True)
