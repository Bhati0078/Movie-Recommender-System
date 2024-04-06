# Streamlit Movie Recommender System

This project implements a movie recommendation system using content-based filtering, leveraging natural language processing (NLP) techniques and cosine similarity. The system is built in Python and utilizes popular libraries such as pandas, numpy, scikit-learn, NLTK, and Streamlit for creating an interactive web application.

The dataset used consists of two CSV files: "tmdb_5000_movies.csv" and "tmdb_5000_credits.csv". These files contain information about movies, including their titles, genres, keywords, cast, crew, and other relevant details.

 ## The workflow involves several steps:

# Data Preprocessing: 
The necessary columns are selected from the datasets and merged. Null values are dropped, and certain columns are renamed for clarity.

# Feature Extraction: 
Information such as genres, keywords, cast, and crew is extracted from the dataset and processed. Text preprocessing techniques like stemming and stop-word removal are applied to clean the data.

# Vectorization:
The textual data is converted into numerical vectors using the CountVectorizer from scikit-learn. This step transforms the text into a matrix of token counts.

# Similarity Calculation: 
Cosine similarity is calculated between the vectors representing different movies. This metric measures the similarity between two movies based on their textual features.

# Recommendation: 
Given a movie title, the system identifies the index of the movie in the dataset, computes its similarity with other movies, and recommends the top 10 most similar movies.

# Streamlit Application:
The recommendation system is deployed as an interactive web application using Streamlit. Users can select a movie from a dropdown menu and receive recommendations along with movie posters fetched from The Movie Database (TMDb) API.

# Model Serialization:
Finally, the processed data and the similarity matrix are serialized using the pickle library for future use.

The recommendation system allows users to input a movie title and receive recommendations based on the content of that movie. This approach focuses on suggesting movies with similar themes, genres, and cast, making it suitable for users interested in discovering movies with comparable attributes.

This project serves as an educational resource for understanding content-based recommendation systems and demonstrates the application of NLP and machine learning techniques in building personalized recommendation engines for movies.# Movie-Recommender-System
