import streamlit as st
import pickle
import pandas as pd
import requests

# Fetch Poster
def fetch_poster(movie_id):
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=e501ef4da1260537c0b0c7adc8827c61'.format(movie_id)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception as e:
        st.error(f"Failed to fetch poster: {e}")
    return None 

# Recommend Movies
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

        recommended_movies = []
        recommended_posters = []

        for i in movie_list:
            movie_id = movies.iloc[i[0]].id
            recommended_movies.append(movies.iloc[i[0]].title)
            poster_url = fetch_poster(movie_id)
            if poster_url:
                recommended_posters.append(poster_url)
            else:
                recommended_posters.append("Poster not available")
                
        return recommended_movies, recommended_posters
    except Exception as e:
        st.error(f"Failed to recommend movies: {e}")
        return [], []

# Load Data
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Streamlit App
st.title('Movie Recommender System')

selected_movie_name = st.selectbox('Select a movie', movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    columns = st.columns(5)
    for col, name, poster in zip(columns, names, posters):
        with col:
            st.text(name)
            st.image(poster)
