import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.sparse.linalg import svds
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
    "faiss": ("faiss_index_274244.index", "17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu")
}

# =============================
# OPTIMIZED LOADING & TRAINING
# =============================
@st.cache_resource(show_spinner=False)
def load_and_prep_data():
    """
    Downloads files, loads data, and pre-trains models once.
    Returns a dictionary containing all necessary assets.
    """
    data_store = {}

    # 1. Download Files
    with st.status("📦 System Initialization...", expanded=True) as status:
        for key, (fname, fid) in FILES.items():
            if not os.path.exists(fname):
                st.write(f"⬇️ Downloading `{fname}`...")
                url = f"https://drive.google.com/uc?id={fid}"
                gdown.download(url, fname, quiet=True)
        status.update(label="✅ Files ready", state="complete")

    # 2. Load Static Data
    data_store["meta"] = pd.read_parquet(FILES["meta"][0])
    data_store["tfidf"] = sparse.load_npz(FILES["tfidf"][0])
    data_store["embed"] = np.load(FILES["embed"][0]).astype("float32")
    data_store["index"] = faiss.read_index(FILES["faiss"][0])

    # 3. Create Mappings
    movie_ids = data_store["meta"]["movie_id"].tolist()
    data_store["movie_ids"] = movie_ids
    data_store["movie_to_idx"] = {m: i for i, m in enumerate(movie_ids)}

    # 4. Prepare Collaborative Filtering (Vectorized)
    df = pd.read_csv(FILES["sample"][0])
    
    # Clean and parse actors more efficiently
    df["actors"] = df["stars"].fillna("").astype(str).str.split(r"[,|]")
    df["movie_id"] = (
        df["title"].astype(str).str.strip() + " (" + df["year"].fillna("NA").astype(str) + ")"
    )

    # Create Interactions
    # Using 'explode' is necessary but we do it once here
    interactions = df.explode("actors")[["actors", "movie_id", "rating"]]
    interactions["actors"] = interactions["actors"].str.strip()
    interactions = interactions[interactions["actors"] != ""]
    
    # Map Users (Actors) and Items (Movies) to Integer IDs
    users = sorted(interactions["actors"].unique())
    items = sorted(interactions["movie_id"].unique())
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: j for j, i in enumerate(items)}
    
    # Create Sparse Matrix for SVD
    # Rows = Users (Actors), Cols = Items (Movies)
    row_ind = interactions["actors"].map(user_to_idx).values
    col_ind = interactions["movie_id"].map(item_to_idx).values
    data_vals = interactions["rating"].values
    
    R_sparse = sparse.csr_matrix(
        (data_vals, (row_ind, col_ind)), 
        shape=(len(users), len(items))
    )

    # 5. Fast Matrix Factorization using SVD (Replacing slow SGD loop)
    # k=20 matches the original K=20
    u_factors, s_vals, vt_factors = svds(R_sparse, k=20)
    
    # Diagonalize S to reconstruct
    s_diag = np.diag(s_vals)
    
    # Store matrices: User Matrix (U @ S) and Item Matrix (Vt)
    # Prediction = (U_user) @ (V_item_T)
    data_store["U"] = u_factors @ s_diag
    data_store["Vt"] = vt_factors  # Already transposed (K x Items)
    
    data_store["user_to_idx"] = user_to_idx
    data_store["items"] = items
    
    # Pre-compute 'seen' items for faster lookup
    data_store["user_seen"] = interactions.groupby("actors")["movie_id"].apply(set).to_dict()

    return data_store

# Initialize System (Runs once)
APP_DATA = load_and_prep_data()

# =============================
# RECOMMENDATION FUNCTIONS
# =============================
def recommend_collab(actor, top_n=10):
    actor = actor.strip()
    idx_map = APP_DATA["user_to_idx"]
    
    if actor not in idx_map:
        return []

    u_idx = idx_map[actor]
    seen = APP_DATA["user_seen"].get(actor, set())

    # Fast Matrix Multiplication: (1, K) @ (K, Items) -> (1, Items)
    user_vector = APP_DATA["U"][u_idx]
    scores = user_vector @ APP_DATA["Vt"]
    
    # Get top indices efficiently
    # We negate scores to use argpartition for "largest"
    top_indices = np.argpartition(-scores, top_n)[:top_n + len(seen)]
    
    # Sort specifically the top candidates
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    recommendations = []
    for idx in top_indices:
        item = APP_DATA["items"][idx]
        if item not in seen:
            recommendations.append((item, float(scores[idx])))
            if len(recommendations) >= top_n:
                break
    
    return recommendations

def recommend_tfidf(movie, n=10):
    idx_map = APP_DATA["movie_to_idx"]
    if movie not in idx_map:
        return []
        
    i = idx_map[movie]
    # Cosine sim calculation
    sims = cosine_similarity(APP_DATA["tfidf"][i], APP_DATA["tfidf"]).flatten()
    sims[i] = -1
    
    # Fast sort top N
    top = np.argpartition(sims, -n)[-n:]
    top = top[np.argsort(sims[top])[::-1]]
    
    return [(APP_DATA["movie_ids"][j], float(sims[j])) for j in top]

def recommend_embed(movie, n=10):
    idx_map = APP_DATA["movie_to_idx"]
    if movie not in idx_map:
        return []
        
    i = idx_map[movie]
    # FAISS search
    D, I = APP_DATA["index"].search(APP_DATA["embed"][i:i+1], n+1)
    
    return [
        (APP_DATA["movie_ids"][j], float(d)) 
        for d, j in zip(D[0], I[0]) if j != i
    ][:n]

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title(f"🎬 Movie Recommendation System (Group {GROUP_ID})")

st.sidebar.header("🔍 Engine Settings")

mode = st.sidebar.radio(
    "Select Model Strategy",
    [
        "📄 Content-Based (TF-IDF)",
        "🧠 Semantic Search (Embeddings)",
        "👥 Collaborative (Actor History)"
    ]
)

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Main Logic
recs = []

try:
    if mode.startswith("📄"):
        movie = st.selectbox("Select a Movie", APP_DATA["movie_ids"])
        if st.button("Recommend", type="primary"):
            recs = recommend_tfidf(movie, top_n)

    elif mode.startswith("🧠"):
        movie = st.selectbox("Select a Movie", APP_DATA["movie_ids"])
        if st.button("Recommend", type="primary"):
            recs = recommend_embed(movie, top_n)

    else:
        # Sort actors list for better UX
        sorted_actors = sorted(APP_DATA["user_to_idx"].keys())
        actor = st.selectbox("Select an Actor", sorted_actors)
        if st.button("Recommend", type="primary"):
            recs = recommend_collab(actor, top_n)

   
   # Display Results
    st.divider()
    if recs:
        st.subheader("Top Recommendations")
        df_recs = pd.DataFrame(recs, columns=["Movie Title", "Similarity Score"])
        
        # Use Streamlit's native column configuration for formatting
        st.dataframe(
            df_recs,
            column_config={
                "Similarity Score": st.column_config.NumberColumn(format="%.4f")
            },
            use_container_width=True,
            hide_index=True
        )
    elif st.session_state.get("button_clicked", False):
        st.info("No recommendations found.")

except Exception as e:
    st.error(f"An error occurred: {e}")
