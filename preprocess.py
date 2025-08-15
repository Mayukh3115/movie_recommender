import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer

def parse_features(x):
    return [i['name'] for i in ast.literal_eval(x)]

def get_director(x):
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            return i['name']
    return ''

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

def preprocess_movies(csv_path="Movies_relevant.csv"):
    movies = pd.read_csv(csv_path)
    movies.dropna(inplace=True)
    movies.drop_duplicates(subset='title', inplace=True)

    movies['genres'] = movies['genres'].apply(parse_features)
    movies['cast'] = movies['cast'].apply(lambda x: parse_features(x)[:3])
    movies['keywords'] = movies['keywords'].apply(parse_features)
    movies['director'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.lower().split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['keywords']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    movies['tags'] = movies['tags'].apply(stem)

    return movies