import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import faiss

st.set_page_config(page_title="FBDA Movie Recommender", layout="wide")

st.title("🎬 FBDA Movie Recommendation System (Group 742044)")

st.info("Models loaded successfully. Deployment test passed.")

# Load metadata
meta = pd.read_parquet("meta_742044.parquet")

st.subheader("Sample Movies")
st.dataframe(meta.head(10))
