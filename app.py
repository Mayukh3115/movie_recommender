import streamlit as st
from preprocess import preprocess_movies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender & Explorer")

@st.cache_resource
def load_movies():
    return preprocess_movies()

@st.cache_resource
def compute_similarity(movies):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

movies = load_movies()
similarity = compute_similarity(movies)

def recommend(movie):
    movie = movie.lower()
    matches = movies[movies['title'].str.lower() == movie]
    if matches.empty:
        return []
    index = matches.index[0]
    dist = similarity[index]
    top = dist.argsort()[::-1][1:6]
    return movies['title'].iloc[top].tolist()

st.header("Recommend Similar Movies")
movie_input = st.text_input("Enter a movie name:")

if movie_input:
    results = recommend(movie_input)
    if results:
        st.success("You may also like:")
        for title in results:
            st.markdown(f"**{title}**")
    else:
        st.error("Movie not found. Please try another title.")

st.header("Explore the Movie Dataset")
with st.expander("View full dataset"):
    st.dataframe(
        movies[['title', 'genres', 'cast', 'director', 'vote_average']].head(50),
        use_container_width=True
    )
