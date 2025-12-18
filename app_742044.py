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
# DOWNLOAD REQUIRED FILES FROM GOOGLE DRIVE
# =============================
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=False)

with st.spinner("Downloading model files (if first run)..."):
    download_from_drive("1kFkoX7hPQC9TvZWLprIQHFwJ4e8CcBqy", "meta_742044.parquet")
    download_from_drive("1PBUuC8N1XvaPnX8x9dbXy4iT_tNBSA1", "sampled_10001_742044.csv")
    download_from_drive("1yXZ0PeLB9dao-9bsmMSRUNnRc2sHfL7", "svd_model_742044.pkl")
    download_from_drive("11hbrXLMYv8U4yH_-U7QKXwprU_00hjEy", "tfidf_vectorizer_742044.pkl")
    download_from_drive("1YZFmBBFCc3AxKsOpM5ECOC4tBT9GIRMY", "tfidf_matrix_742044.npz")
    download_from_drive("1a506s41J0qenzTSOHg5LVSS0XnL-Fpq", "embeddings_742044.npy")
    download_from_drive("17orGU6B1SMocR2y_Z_b_PKFK7Q_T5yfu", "faiss_index_742044.index")

# =============================
# CONFIG
# =============================
GROUP_ID = "742044"
SEED = 742044

SAMPLED_CSV = f"sampled_10001_{GROUP_ID}.csv"
SVD_MODEL_FILE = f"svd_model_{GROUP_ID}.pkl"
TFIDF_VECTORIZER_FILE = f"tfidf_vectorizer_{GROUP_ID}.pkl"
TFIDF_MATRIX_FILE = f"tfidf_matrix_{GROUP_ID}.npz"
EMBEDDINGS_FILE = f"embeddings_{GROUP_ID}.npy"
FAISS_INDEX_FILE = f"faiss_index_{GROUP_ID}.index"
META_FILE = f"meta_{GROUP_ID}.parquet"

st.set_page_config(page_title=f"Movie Recommender {GROUP_ID}", layout="wide")
st.title(f"🎬 Movie Recommendation System (Group {GROUP_ID})")

# =============================
# LOAD DATA & MODELS (SAFE)
# =============================
@st.cache_data
def load_meta():
    return pd.read_parquet(META_FILE)

@st.cache_resource
def load_artifacts():
    svd = joblib.load(SVD_MODEL_FILE)
    tfidf = joblib.load(TFIDF_VECTORIZER_FILE)
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_FILE)
    emb = np.load(EMBEDDINGS_FILE).astype("float32")
    index = faiss.read_index(FAISS_INDEX_FILE)
    return svd, tfidf, tfidf_matrix, emb, index

try:
    meta = load_meta()
    svd, tfidf, tfidf_matrix, emb, index = load_artifacts()
    st.success("Models and data loaded successfully ✅")
except Exception as e:
    st.error("❌ Failed to load data or models")
    st.exception(e)
    st.stop()

# =============================
# PREP
# =============================
movie_ids = meta["movie_id"].tolist()
movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

# =============================
# BUILD ACTOR → SEEN MOVIES
# =============================
@st.cache_data
def build_user_seen():
    import re
    df = pd.read_csv(SAMPLED_CSV)

    def make_movie_id(row):
        y = row["year"]
        y_str = "NA" if pd.isna(y) else str(int(float(y)))
        return f'{str(row["Title"]).strip()} ({y_str})'

    def parse_actors(stars_str):
        parts = re.split(r"[,\|]", str(stars_str))
        return [p.strip() for p in parts if p.strip()]

    df["movie_id"] = df.apply(make_movie_id, axis=1)
    df["actors"] = df["stars"].apply(parse_actors)

    inter = df[["movie_id", "rating", "actors"]].explode("actors")
    inter = inter.rename(columns={"actors": "user_id", "movie_id": "item_id"})
    inter = inter[inter["user_id"].astype(str).str.len() > 0]

    user_seen = inter.groupby("user_id")["item_id"].apply(set).to_dict()
    actors = sorted(user_seen.keys())

    return user_seen, actors

user_seen, actors = build_user_seen()

# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_tfidf(movie_id, top_n=10):
    i = movie_to_idx.get(movie_id)
    if i is None:
        return []

    sims = linear_kernel(tfidf_matrix[i], tfidf_matrix).flatten()
    sims[i] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(movie_ids[j], float(sims[j])) for j in top_idx]

def recommend_embed(movie_id, top_n=10):
    i = movie_to_idx.get(movie_id)
    if i is None:
        return []

    q = emb[i:i+1]
    scores, idxs = index.search(q, top_n + 1)

    out = []
    for score, j in zip(scores[0], idxs[0]):
        if int(j) == i:
            continue
        out.append((movie_ids[int(j)], float(score)))
        if len(out) >= top_n:
            break
    return out

def recommend_svd(actor_name, top_n=10, candidate_pool=2000):
    actor_name = actor_name.strip()
    seen = user_seen.get(actor_name, set())

    rng = np.random.default_rng(SEED)
    unseen = [m for m in movie_ids if m not in seen]
    if not unseen:
        return []

    if len(unseen) > candidate_pool:
        candidates = rng.choice(unseen, size=candidate_pool, replace=False).tolist()
    else:
        candidates = unseen

    scored = []
    for item in candidates:
        try:
            est = svd.predict(actor_name, item).est
        except:
            est = 0
        scored.append((item, float(est)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

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
top_n = st.sidebar.slider("Top N Recommendations", 5, 20, 10)

if mode.startswith("Content-Based"):
    movie = st.selectbox("Select a movie", movie_ids)
    recs = recommend_tfidf(movie, top_n)

elif mode.startswith("Text Embeddings"):
    movie = st.selectbox("Select a movie", movie_ids)
    recs = recommend_embed(movie, top_n)

else:
    actor = st.selectbox("Select an actor", actors)
    recs = recommend_svd(actor, top_n)

# =============================
# OUTPUT
# =============================
st.subheader("Recommended Movies")

rows = []
for mid, score in recs:
    row_df = meta[meta["movie_id"] == mid]
    if row_df.empty:
        continue

    r = row_df.iloc[0].to_dict()
    r["score"] = score
    rows.append(r)

if not rows:
    st.warning("No recommendations found.")
else:
    out_df = pd.DataFrame(rows)[
        ["movie_id", "year", "genre", "rating", "votes", "score", "stars"]
    ]

    st.dataframe(out_df, use_container_width=True)

    with st.expander("Show Descriptions"):
        for row in rows:
            st.markdown(
                f"**{row['movie_id']}**  \n"
                f"Score: `{row['score']:.4f}`  \n"
                f"{row['description']}"
            )
            st.markdown("---")

