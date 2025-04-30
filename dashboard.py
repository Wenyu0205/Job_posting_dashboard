# ---- Import libraries ----
import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load Data ----
with open('Linkedin_jobs_with_extracted_skills.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ---- Prepare Skills ----
df['skills_list'] = df['extracted_skills'].apply(lambda x: [skill.strip() for skill in x.split(',')] if pd.notna(x) else [])

# ---- Load Embedding Model ----
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- Embed all Skills ----
all_skills = set(skill for skills in df['skills_list'] for skill in skills)
skill_to_vec = {skill: model.encode(skill) for skill in all_skills}

# ---- Compute Job Embeddings ----
def aggregate_job_embedding(skills):
    skill_vecs = [skill_to_vec[skill] for skill in skills if skill in skill_to_vec]
    if skill_vecs:
        return np.mean(skill_vecs, axis=0)
    else:
        return np.zeros(model.get_sentence_embedding_dimension())

df['job_embedding'] = df['skills_list'].apply(aggregate_job_embedding)

job_titles = df['jobtitle'].tolist()
job_vectors = np.vstack(df['job_embedding'].values)

# ---- Precompute Top 5 Similar Jobs for each Job ----
similarity_matrix = cosine_similarity(job_vectors)

top_5_jobs = []
top_5_scores = []

for i in range(len(job_titles)):
    sims = similarity_matrix[i]
    top_indices = np.argsort(sims)[::-1][1:6]  # exclude self
    top_5_jobs.append([job_titles[j] for j in top_indices])
    top_5_scores.append([sims[j] for j in top_indices])

df['top_5_similar_jobs'] = top_5_jobs
df['top_5_similar_scores'] = top_5_scores

# ---- Streamlit App ----
st.set_page_config(page_title="Job & Skill Matching Dashboard", layout="wide")
st.title("🎯 Job & Skill Matching Dashboard (Live Embedding)")

tab1, tab2 = st.tabs(["Search by Job Title", "Search by Skill"])

with tab1:
    job_select = st.selectbox("Select a Job Title:", [""] + sorted(job_titles))
    job_input = st.text_input("Or type a Job Title manually:")

    final_job_input = job_input.strip() if job_input.strip() else job_select

    st.markdown("---")

    if final_job_input:
        match = df[df['jobtitle'] == final_job_input]
        if not match.empty:
            row = match.iloc[0]
            st.subheader(f"📄 Top 5 Jobs Similar to: {row['jobtitle']}")

            similar_jobs = row['top_5_similar_jobs']
            similar_scores = row['top_5_similar_scores']

            for i, (title, score) in enumerate(zip(similar_jobs, similar_scores), start=1):
                job_row = df[df['jobtitle'] == title]
                if not job_row.empty:
                    desc = job_row.iloc[0].get('description', 'No description available.')
                    skills = job_row.iloc[0].get('extracted_skills', 'No skills available.')

                    with st.expander(f"🔹 {i}. {title} (Similarity: {score:.3f})"):
                        st.markdown(f"**📈 Similarity Score:** `{score:.3f}`")
                        st.markdown(f"**📝 Description:** {desc}")
                        st.markdown(f"**🛠️ Extracted Skills:** {skills}")
                else:
                    with st.expander(f"🔹 {i}. {title} (Similarity: {score:.3f})"):
                        st.warning("No additional information available for this job.")
        else:
            st.warning("Job title not found.")

with tab2:
    skill_input = st.text_input("Enter a Skill:")
    st.markdown("---")

    if skill_input:
        skill_vec = model.encode(skill_input).reshape(1, -1)

        similarities = cosine_similarity(skill_vec, job_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:5]

        st.subheader(f"📄 Top 5 Jobs Relevant to Skill: {skill_input}")
        for i, idx in enumerate(top_indices, start=1):
            job_title = job_titles[idx]
            similarity_score = similarities[idx]

            job_row = df[df['jobtitle'] == job_title]
            if not job_row.empty:
                desc = job_row.iloc[0].get('description', 'No description available.')
                skills = job_row.iloc[0].get('extracted_skills', 'No skills available.')

                with st.expander(f"🔹 {i}. {job_title} (Similarity: {similarity_score:.3f})"):
                    st.markdown(f"**📈 Similarity Score:** `{similarity_score:.3f}`")
                    st.markdown(f"**📝 Description:** {desc}")
                    st.markdown(f"**🛠️ Extracted Skills:** {skills}")
            else:
                with st.expander(f"🔹 {i}. {job_title} (Similarity: {similarity_score:.3f})"):
                    st.warning("No additional information available for this job.")
