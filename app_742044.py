import streamlit as st
st.title("FBDA Movie Recommendation System")
st.write("App loaded successfully")
import gdown
import os

def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# 🔽 DOWNLOAD LARGE FILES FROM GOOGLE DRIVE
download_from_drive("1yXZQPeLB9dao-9bsnM1SRUNnRc2sWfL7", "svd_model_742044.pkl")
download_from_drive("1h1brxLMYv8u4yH-_U70Kxwprv_u0hjEy", "tfidf_vectorizer_742044.pkl")
download_from_drive("1YZfmBFBCc3AxKsoPm5E0CQ4tBT9GIRMY", "tfidf_matrix_742044.npz")
download_from_drive("1a506s4IJqeunzTSQHg5LVSS0xnLl-Fpq", "embeddings_742044.npy")
download_from_drive("17orGU6B1SMocR2y_Z_b_PKFK7O_TSyfu", "faiss_index_742044.index")
download_from_drive("1kfK0x7hPQC9TvZwLprlQHFwJ4e8CcBqy", "meta_742044.parquet")
download_from_drive("1PBUEUc8N1XvaPnX8x9dbXy4iT_tNBSAl", "sampled_10001_742044.csv")
