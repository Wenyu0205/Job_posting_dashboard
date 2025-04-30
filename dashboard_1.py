# ---- Import libraries ----
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# ---- DEBUG: Check working directory ----
print("🟡 Files in directory:", os.listdir())

# ---- Load Model ----
print("🟡 Loading model...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()
print("✅ Model loaded.")

# ---- Load Job Data ----
print("🟡 Loading job data...")
def load_job_data():
    df = pd.read_csv('job_embeddings_4.csv')
    embed_cols = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
    job_vectors = df[embed_cols].values
    return df, job_vectors
job_df, job_vectors = load_job_data()
print("✅ Job data loaded:", job_df.shape)

# ---- Load Skill Embeddings ----
print("🟡 Loading skill embeddings...")
def load_skill_embeddings():
    df = pd.read_csv('skill_embeddings_1.csv')
    skill_vec_dict = {row['skill']: row.drop('skill').values for _, row in df.iterrows()}
    return skill_vec_dict
skill_vec_dict = load_skill_embeddings()
print("✅ Skill embeddings loaded:", len(skill_vec_dict))

# ---- Basic Sanity Check ----
print("🟡 Columns in job_df:", job_df.columns.tolist())
print("🟡 First 2 rows:\n", job_df.head(2))

job_titles = job_df['jobtitle'].dropna().unique().tolist()
skill_options = sorted(skill_vec_dict.keys())

# ---- Streamlit Layout ----
st.set_page_config(page_title="Job & Skill Matching Dashboard", layout="wide")
st.title("🎯 Job & Skill Matching Dashboard")

tab1, tab2 = st.tabs(["🔎 Search by Job", "🛠️ Search by Skill(s)"])

# ---- Job-Based Search ----
with tab1:
    job_select = st.selectbox("Select a Job Title:", [""] + sorted(job_titles))
    job_input = st.text_input("Or type a Job Title manually:")

    final_input = job_input.strip() if job_input.strip() else job_select
    st.markdown("---")

    if final_input:
        match = job_df[job_df['jobtitle'] == final_input]
        if not match.empty:
            idx = match.index[0]
            query_vec = job_vectors[idx].reshape(1, -1)
            similarities = cosine_similarity(query_vec, job_vectors)[0]
            top_indices = np.argsort(similarities)[::-1][1:11]

            st.subheader(f"📄 Top 10 Jobs Similar to: {final_input}")
            for i, idx in enumerate(top_indices, 1):
                row = job_df.iloc[idx]
                st.markdown(f"### 🔹 {i}. {row['jobtitle']} (Similarity: `{similarities[idx]:.3f}`)")
                st.markdown("**📝 Description:**", unsafe_allow_html=True)
                st.markdown(row['description'], unsafe_allow_html=True)
                st.markdown(f"**🛠️ Skills:** {row['extracted_skills']}")
                st.markdown("---")
        else:
            st.warning("Job title not found.")

# ---- Skill-Based Search ----
with tab2:
    st.markdown("💡 You can either manually type skills, select them from the list below, or both.")

    # Manual text input
    skill_input = st.text_input("Enter one or more skills (comma-separated):")

    # Dropdowns for selecting skills
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        skill_1 = st.selectbox("Skill 1", [""] + skill_options)
    with col2:
        skill_2 = st.selectbox("Skill 2", [""] + skill_options)
    with col3:
        skill_3 = st.selectbox("Skill 3", [""] + skill_options)
    with col4:
        skill_4 = st.selectbox("Skill 4", [""] + skill_options)
    with col5:
        skill_5 = st.selectbox("Skill 5", [""] + skill_options)

    st.markdown("---")

    # Combine both input sources
    manual_skills = [s.strip() for s in skill_input.split(',') if s.strip()]
    dropdown_skills = [s for s in [skill_1, skill_2, skill_3, skill_4, skill_5] if s]
    combined_skills = manual_skills + dropdown_skills

    if combined_skills:
        found_vecs = [skill_vec_dict[s] for s in combined_skills if s in skill_vec_dict]

        if found_vecs:
            input_vec = np.mean(found_vecs, axis=0).reshape(1, -1)
            similarities = cosine_similarity(input_vec, job_vectors)[0]
            top_indices = np.argsort(similarities)[::-1][:10]

            st.subheader(f"📄 Top 10 Jobs Relevant to Skill(s): `{', '.join(combined_skills)}`")
            for i, idx in enumerate(top_indices, 1):
                row = job_df.iloc[idx]
                st.markdown(f"### 🔹 {i}. {row['jobtitle']} (Similarity: `{similarities[idx]:.3f}`)")
                st.markdown("**📝 Description:**", unsafe_allow_html=True)
                st.markdown(row['description'], unsafe_allow_html=True)
                st.markdown(f"**🛠️ Skills:** {row['extracted_skills']}")
                st.markdown("---")
        else:
            st.warning("No valid skills recognized. Check your spelling or try different terms.")


