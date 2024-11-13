import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from typing import List


movies_data = pd.read_csv('movies_dataset_reducido.csv', low_memory=False)
credits_data = pd.read_csv('credits_reducido.csv', low_memory=False)


print("Movies Data Shape:", movies_data.shape)
print("Credits Data Shape:", credits_data.shape)


print("\nMovies Data Head:")
print(movies_data.head())
print("\nCredits Data Head:")
print(credits_data.head())


print("\nMovies Data Info:")
movies_data.info()
print("\nCredits Data Info:")
credits_data.info()


movies_data_cleaned = movies_data.dropna(subset=['release_date'])


movies_data_cleaned['return'] = np.where(
    (movies_data_cleaned['budget'] > 0),
    movies_data_cleaned['revenue'] / movies_data_cleaned['budget'],
    0)


movies_data_cleaned['release_date'] = pd.to_datetime(movies_data_cleaned['release_date'], errors='coerce')
movies_data_cleaned['release_year'] = movies_data_cleaned['release_date'].dt.year


movies_data_cleaned['budget'] = pd.to_numeric(movies_data_cleaned['budget'], errors='coerce')
movies_data_cleaned['revenue'] = pd.to_numeric(movies_data_cleaned['revenue'], errors='coerce')
movies_data_cleaned['return'] = np.where(movies_data_cleaned['budget'] > 0, movies_data_cleaned['revenue'] / movies_data_cleaned['budget'], 0)


columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
movies_data_cleaned = movies_data_cleaned.drop(columns=columns_to_drop)


def desanidar_columna(columna):
    try:
        return ast.literal_eval(columna) if isinstance(columna, str) else []
    except ValueError:
        return []

movies_data_cleaned['belongs_to_collection'] = movies_data_cleaned['belongs_to_collection'].apply(desanidar_columna)
movies_data_cleaned['production_companies'] = movies_data_cleaned['production_companies'].apply(desanidar_columna)
movies_data_cleaned['production_countries'] = movies_data_cleaned['production_countries'].apply(desanidar_columna)


print("\nMovies Data Cleaned Info:")
print(movies_data_cleaned.info())
print("\nMovies Data Cleaned Head:")
print(movies_data_cleaned.head())


movies_data = movies_data.merge(credits_data, on='id')


movies_data['overview'] = movies_data['overview'].fillna('')


def obtener_informacion_principal(x, key):
    try:
        items = ast.literal_eval(x)
        return ' '.join([d[key] for d in items if key in d])
    except:
        return ''


movies_data['actors'] = movies_data['cast'].apply(lambda x: obtener_informacion_principal(x, 'name'))
movies_data['director'] = movies_data['crew'].apply(lambda x: obtener_informacion_principal(x, 'name') if 'Director' in x else '')
movies_data['genres'] = movies_data['genres'].apply(lambda x: obtener_informacion_principal(x, 'name'))
movies_data['production_companies'] = movies_data['production_companies'].apply(lambda x: obtener_informacion_principal(x, 'name'))


movies_data['content'] = movies_data['overview'] + ' ' + movies_data['genres'] + ' ' + \
                         movies_data['actors'] + ' ' + movies_data['director'] + ' ' + \
                         movies_data['production_companies']


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_data.index, index=movies_data['title']).drop_duplicates()


def recomendar_peliculas(title, cosine_sim=cosine_sim, df=movies_data, indices=indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


pelicula = "Toy Story"
print(f"Películas recomendadas para '{pelicula}':\n")
print(recomendar_peliculas(pelicula))


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/recomendar/{title}")
def get_recommendations(title: str):
    try:
        recomendaciones = recomendar_peliculas(title)
        return {"title": title, "recommendations": recomendaciones}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
