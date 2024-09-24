import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st
import re
import time
import gdown

st.set_page_config(page_title="TriStep - Career and Learning Recommendation System", page_icon="🚀", layout="wide")

def preprocess_text_simple(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\*+', '', text)
    return text

def remove_asterisks(text):
    if pd.isna(text):
        return text
    return re.sub(r'\*+', '', text)

def recommend_job(user_input, df, vectorizer, tfidf_matrix, experience_levels, work_types, name):
    filtered_df = df.copy()
    if experience_levels:
        filtered_df = filtered_df[filtered_df['formatted_experience_level'].isin(experience_levels)]
    if work_types:
        filtered_df = filtered_df[filtered_df['formatted_work_type'].isin(work_types)]
    if name and name != 'All':
        filtered_df = filtered_df[filtered_df['name'] == name]
    
    if filtered_df.empty:
        return None

    user_input_processed = preprocess_text_simple(user_input)
    user_tfidf = vectorizer.transform([user_input_processed])
    
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix[filtered_df.index]).flatten()
    
    above_zero = cosine_similarities > 0
    if not any(above_zero):
        return None

    threshold = np.percentile(cosine_similarities[above_zero], 95)
    
    above_threshold = cosine_similarities >= threshold
    top_course_indices = np.where(above_threshold)[0]
    
    top_course_indices = top_course_indices[np.argsort(cosine_similarities[top_course_indices])[::-1]]
    
    top_courses = filtered_df.iloc[top_course_indices].copy()
    top_courses.reset_index(drop=True, inplace=True)
    
    top_courses['cosine_similarity'] = cosine_similarities[top_course_indices]
    
    return top_courses

def recommend_course(user_input, df, vectorizer, tfidf_matrix):
    start_time = time.time()
    user_input_processed = preprocess_text_simple(user_input)
    user_tfidf = vectorizer.transform([user_input_processed])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_course_indices = cosine_similarities.argsort()[::-1]
    recommendation_time = time.time() - start_time
    recommendations = df.iloc[top_course_indices].copy()
    recommendations['cosine_similarity'] = cosine_similarities[top_course_indices]
    return recommendations

def imdb_score(df, q=0.95):
    df = df.copy()
    m = df['Number of viewers'].quantile(q)
    c = (df['Rating'] * df['Number of viewers']).sum() / df['Number of viewers'].sum()
    df["score"] = df.apply(lambda x: (x.Rating * x['Number of viewers'] + c*m) / (x['Number of viewers'] + m), axis=1)
    return df

def change_page(direction):
    if 'page' in st.session_state:
        st.session_state.page += direction
    else:
        st.session_state.page = 0

def convert_to_yearly(salary, pay_period):
    try:
        salary = float(salary)
        
        if pay_period == 'YEARLY':
            return salary
        elif pay_period == 'MONTHLY':
            return salary * 12
        elif pay_period == 'HOURLY':
            return salary * 40 * 52
        else:
            return 'Unknown'
    except (ValueError, TypeError):
        return 'Unknown'

def preprocess_salary(df):
    df['min_salary'] = df.apply(lambda x: convert_to_yearly(x['min_salary'], x['pay_period']), axis=1)
    df['max_salary'] = df.apply(lambda x: convert_to_yearly(x['max_salary'], x['pay_period']), axis=1)
    return df

@st.cache_data
def load_job_data():
    url = 'https://drive.google.com/uc?export=download&id=1fBOB-dm_BJasoJfA_CwUFMBINwgvWPeW'
    output = 'linkedin.csv'
    gdown.download(url, output, quiet=False)
    df_job = pd.read_csv(output)
    df_job['Combined'] = df_job['title'].fillna('') + ' ' + df_job['description_x'].fillna('') + ' ' + df_job['skills_desc'].fillna('')
    df_job['Combined'] = df_job['Combined'].apply(preprocess_text_simple)
    df_job['title'] = df_job['title'].apply(remove_asterisks)
    vectorizer_job = TfidfVectorizer(stop_words='english')
    tfidf_matrix_job = vectorizer_job.fit_transform(df_job['Combined'])
    return df_job, vectorizer_job, tfidf_matrix_job

@st.cache_data
def load_course_data():
    url = 'https://drive.google.com/uc?id=1tnpLFGqbmGRU_EDxUpuMCupx4-HJxEqF'
    output = 'Online_Courses.csv'
    gdown.download(url, output, quiet=False)

    df_course = pd.read_csv(output)
    df_course.drop(columns=['Unnamed: 0','Program Type', 'Courses', 'Level', 'Number of Reviews',
           'Unique Projects', 'Prequisites', 'What you learn', 'Related Programs',
           'Monthly access', '6-Month access', '4-Month access', '3-Month access',
           '5-Month access', '2-Month access', 'School', 'Topics related to CRM',
           'ExpertTracks', 'FAQs', 'Course Title', 'Course URL',
           'Course Short Intro', 'Weekly study', 'Premium course',
           "What's include", 'Rank', 'Created by', 'Program'], inplace=True)
    df_course = df_course.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df_course = df_course.drop_duplicates(subset=['Title', 'Short Intro'])
    translations = {
        '计算机科学': 'Computer Science',
        'Ciencia de Datos': 'Data Science',
        'Negocios': 'Business',
        'Ciencias de la Computación': 'Computer Science',
        'Negócios': 'Business',
        'データサイエンス': 'Data Science',
        'Tecnologia da informação': 'Information Technology'
    }
    df_course['Category'] = df_course['Category'].replace(translations)
    df_course['Rating'] = df_course['Rating'].str.replace('stars', '', regex=False)
    df_course['Number of viewers'] = df_course['Number of viewers'].str.replace(r'\D+', '', regex=True)
    df_course['combined'] = df_course['Title'] + ' ' + df_course['Short Intro'].fillna('') + ' ' + df_course['Skills'].fillna('') + ' ' + df_course['Category'].fillna('') + ' ' + df_course['Sub-Category'].fillna('')
    df_course['combined'] = df_course['combined'].apply(preprocess_text_simple)
    df_course = df_course.fillna('Unknown')
    df_course['Number of viewers'] = pd.to_numeric(df_course['Number of viewers'], errors='coerce').fillna(0).astype(int)
    df_course['Rating'] = pd.to_numeric(df_course['Rating'], errors='coerce').fillna(0)
    df_course['Subtitle Languages'] = df_course['Subtitle Languages'].str.replace('Subtitles: ', '', regex=False)
    
    keywords = ['Participant', 'Designed', 'Learners', 'prior', 'experience', 'natural', 'space', 'aeronautics']
    
    def remove_keywords(text, keywords):
        if pd.isna(text):
            return np.nan
        if any(keyword in text for keyword in keywords):
            return np.nan
        return text
    
    df_course['Subtitle Languages'] = df_course['Subtitle Languages'].apply(lambda x: remove_keywords(x, keywords))
    
    vectorizer_course = TfidfVectorizer(stop_words='english')
    tfidf_matrix_course = vectorizer_course.fit_transform(df_course['combined'])
    return df_course, vectorizer_course, tfidf_matrix_course

@st.cache_data
def download_images():
    url1 = 'https://drive.google.com/uc?id=1lhfFczKatGDEuq3ux2y-AqfPpVC96UZ9'
    output1 = 'Minimalist_Black_and_White_Blank_Paper_Document_1.png'
    gdown.download(url1, output1, quiet=False)

    url2 = 'https://drive.google.com/uc?id=1hbpQIE7Ez0Z4k1Sfq8FSO80_5HRujdjP'
    output2 = 'nobg2.png'
    gdown.download(url2, output2, quiet=False)

    return output1, output2

# Load data
df_job, vectorizer_job, tfidf_matrix_job = load_job_data()
df_job = preprocess_salary(df_job)
df_job.fillna("Unknown", inplace=True)
df_course, vectorizer_course, tfidf_matrix_course = load_course_data()

image1_path, image2_path = download_images()

# New role selection page
def show_user_page():
    # Original page logic for the User
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    st.sidebar.image(image1_path, use_column_width=True)
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", ('🏢 Home', '📊 Step 1: Explore', '💼 Step 2: Find', '📚 Step 3: Grow'))
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 TriStep 🚀")
    st.sidebar.markdown("Created By M-Tree")

    # Place the existing logic for different pages here, e.g.:
    if page == '🏢 Home':
        st.title("🏢 Home")
        st.write("This is the Home page content.")
        # Add other pages' content based on the existing logic.

def show_contributor_page():
    st.title("Contributor Page")
    st.write("This page is for contributors. It is under construction.")

# Main role selection
def main():
    st.title("Welcome to TriStep Platform")

    role = st.selectbox("Choose your role:", ["User", "Contributor"])

    if role == "User":
        show_user_page()  # Show the user page
    elif role == "Contributor":
        show_contributor_page()  # Show the contributor page

if __name__ == "__main__":
    main()
