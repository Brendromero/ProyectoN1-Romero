from fastapi import FastAPI, HTTPException
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
from typing import Any, Optional, List, Dict
from ast import literal_eval
import ast
from sklearn.neighbors import NearestNeighbors
import os
import uvicorn

app = FastAPI()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

app.title = 'Juegos de Stream'
app.description = 'Proyecto Steam Games'
app.contact = {'name': 'Brendaromero', 'url': 'https://github.com/Brendromero', 'email': 'brendromerok@gmail.com'}

df = pd.read_csv('./new_steam_games.csv')
df_model = pd.read_csv('modelo_aprendizaje.csv')




# Calcular la cantidad de items y el porcentaje de contenido Free por cada año para un desarrollador dado
def calcular_estadisticas_desarrollador(desarrollador):
    # Convertir el nombre del desarrollador a mayúsculas
    desarrollador = desarrollador.lower()

    # Filtrar el DataFrame por el desarrollador dado
    df_desarrollador = df[df['developer'].str.lower() == desarrollador]

    # Verificar si se encontraron datos para el desarrollador
    if df_desarrollador.empty:
        raise HTTPException(status_code=404, detail="Desarrollador no encontrado")

    # Calcular la cantidad total de items y el porcentaje de contenido Free por cada año
    estadisticas_por_anio = {}
    for year, group in df_desarrollador.groupby(df_desarrollador['release_date'].str.slice(0, 4)):
        total_items = group.shape[0]
        free_items = group[group['price'] == 'Free'].shape[0]
        percent_free = (free_items / total_items) * 100 if total_items > 0 else 0
        estadisticas_por_anio[year] = {"total_items": total_items, "free_percent": percent_free}

    return estadisticas_por_anio

@app.get('/Desarrollador/', tags=['General'])
async def developer(desarrollador: str):
    """Ingrese un desarrollador para obtener la cantidad de items y porcentaje de contenido Free por cada año"""
    try:
        estadisticas = calcular_estadisticas_desarrollador(desarrollador)
        return estadisticas
    except HTTPException as e:
        raise e
    
    "***"
    
def calcular_datos_usuario(user_id):
    # Convertir el ID de usuario a minúsculas
    user_id_lower = user_id.lower()
    
    # Filtrar el DataFrame por el ID de usuario dado (en minúsculas)
    user_data = df[df['user_id'].str.lower() == user_id_lower]

    # Verificar si se encontraron datos para el usuario
    if user_data.empty:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Calcular la cantidad de dinero gastado por el usuario
    total_money_spent = user_data['price'].sum()

    # Calcular el porcentaje de recomendación
    recommend_percent = user_data['recommend'].mean() * 100

    # Calcular la cantidad de items
    total_items = user_data['items_count'].sum()

    return {
        "total_money_spent": total_money_spent,
        "recommend_percent": recommend_percent,
        "total_items": total_items
    }
    

@app.get('/Datos de Usuario/{user_id}', tags=['General'])
async def userdata(user_id: str):
    """Ingrese el id de usuario para obtener la cantidad de dinero gastado por el mismo, el porcentaje de recomendación y cantidad de items."""
    try:
        datos_usuario = calcular_datos_usuario(user_id)
        return datos_usuario
    except HTTPException as e:
        raise e

    
"***"

# Convierto las cadenas de la columna 'genres' en listas reales de Python utilizando ast.literal_eval()
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        return np.nan

# Convierte los datos de la columna 'genres' en cadenas de texto
df['genres'] = df['genres'].astype(str)

# Separa los géneros y crea nuevas filas para cada uno
df_expanded = df.assign(genres=df['genres'].str.strip('[]').str.split(', ')).explode('genres')

# Elimina espacios en blanco alrededor de los géneros
df_expanded['genres'] = df_expanded['genres'].str.strip()

@app.get('/Usuario por genero/', tags=['General'])
def userforgenre(genero: str):
    """Ingrese el género para obtener el usuario que acumula mas horas de jugadas por dicho género y una lista de acumulación de horas jugadas por año de lanzamiento."""
    
    # Convertir el género ingresado a minúsculas
    genero_lower = genero.lower()
    
    # Filtrar el DataFrame por el género dado
    df_genre = df_expanded[df_expanded['genres'].apply(lambda x: isinstance(x, str) and genero_lower in x.lower())]

    # Verificar si se encontraron datos para el género dado
    if df_genre.empty:
        raise HTTPException(status_code=404, detail="Género no encontrado")
    
    # Se toma las columnas de los DataFrame y usamos el dropna para las columnas 'genres y 'pplaytime_forever'
    generos = df_expanded[['item_id', 'user_id', 'release_date', 'playtime_forever', 'genres']]
    generos = generos.dropna(subset=['genres', 'playtime_forever'])

    # Filtro por el género específico
    generos = generos[generos['genres'] == genero]

    # Obtener el usuario que acumula más horas de juego
    user_max_playtime = df_genre.loc[df_genre['playtime_forever'].idxmax()]['user_id']

    # Crear una lista de acumulación de horas jugadas por año de lanzamiento
    playtime_by_year = df_genre.groupby(df_genre['release_date'].str.slice(0, 4))['playtime_forever'].sum().to_dict()

    return {
        "user_max_playtime": user_max_playtime,
        "playtime_by_year": playtime_by_year
    }
    
"***"

@app.get('/Mejores desarrolladores por anio', tags=['General'])
def best_developer_year(anio: int):
    """Ingrese el año para obtener el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado."""
    # Verificar que el año sea un entero
    if not isinstance(anio, int):
        raise HTTPException(status_code=400, detail="El año debe ser un entero.")
    
    # Seleccionar las columnas necesarias
    usuario = df[['item_id', 'user_id', 'recommend', 'release_date', 'developer']].copy()

    # Convertir 'release_date' a datetime y filtrar por año
    usuario['release_date'] = pd.to_datetime(usuario['release_date'], errors='coerce')
    usuario = usuario[usuario['release_date'].dt.year == anio]

    # Verificar si el DataFrame filtrado está vacío
    if usuario.empty:
        raise HTTPException(status_code=404, detail="El año ingresado es incorrecto o no se han encontrado datos.")

    # Filtrar por juegos recomendados
    juegos_recomendados = usuario[usuario['recommend'] == True]

    # Agrupar por desarrollador y contar la cantidad de juegos recomendados
    developer_counts = juegos_recomendados['developer'].value_counts().reset_index()
    developer_counts.columns = ['developer', 'recommended_games']

    # Ordenar por la cantidad de juegos recomendados en orden descendente
    developer_counts = developer_counts.sort_values(by='recommended_games', ascending=False).reset_index(drop=True)

    # Tomar los top 3 desarrolladores
    top_3_developers = developer_counts.head(3)

    return top_3_developers.to_dict(orient='records')

"***"

@app.get('/Analisis de sentimiento/', tags=['General'])
def developer_reviews_analysis(desarrollador: str):
    """Ingrese el desarrollador para obtener un diccionario con el nombre del mismo y una lista con la cantidad total de registros de reseñas de usuarios."""
    try:
        # Convertir el nombre del desarrollador a minúsculas para asegurar la consistencia
        desarrollador = desarrollador.lower()

        # Filtrar las reseñas para el desarrollador proporcionado
        desarrollador_reviews = df[df['developer'].str.lower() == desarrollador].copy()

        # Verificar si el DataFrame filtrado está vacío
        if desarrollador_reviews.empty:
            raise HTTPException(status_code=404, detail="No se ha encontrado ningún desarrollador con ese nombre.")

        # Contar las reseñas positivas y negativas
        positive_reviews = len(desarrollador_reviews[desarrollador_reviews['sentiment_analysis'] == 2])
        negative_reviews = len(desarrollador_reviews[desarrollador_reviews['sentiment_analysis'] == 0])

        # Devolver el resultado en un diccionario
        result = {
            'Desarrollador': desarrollador,
            'Reseñas Positivas': positive_reviews,
            'Reseñas Negativas': negative_reviews
        }

        return result
    except Exception as e:
        # Manejar cualquier excepción inesperada y devolver un mensaje de error genérico
        return {'Error': f'Ha ocurrido un error al procesar la solicitud: {str(e)}'}
    
    "***"
    
@app.get('/Recomendaciones/', tags=['Modelo'])
def recomendacion_usuario(user_id: str):
    """Ingresa un id de usuario para obtener una lista de 5 juegos recomendados para dicho usuario."""
    try:
        # Convierto user_id a minúsculas
        user_id = user_id.lower()
        
        if not isinstance(user_id, str):
            raise ValueError('El ID de usuario debe ser un string.')

        # Eliminar espacios en blanco adicionales alrededor del ID de usuario
        user_id = user_id.strip()

        # Filtrar datos del usuario específico
        user_data = df_model[df_model['user_id'].str.lower() == user_id]

        if user_data.empty:
            return {'Error': f'No se encontraron registros para el usuario {user_id}.'}

        # Crear un modelo de vecinos más cercanos con Nearest Neighbors
        model = NearestNeighbors(metric='cosine', algorithm='brute')

        # Crear una matriz de usuario-elemento
        user_item_matrix = pd.crosstab(df_model['user_id'], df_model['item_id'])

        # Ajustar el modelo
        model.fit(user_item_matrix.values)

        # Obtener los juegos más similares
        user_games = user_item_matrix.loc[user_id, :]
        distances, indices = model.kneighbors(user_games.values.reshape(1, -1), n_neighbors=6) # 6 para incluir al propio usuario

        # Seleccionar los juegos recomendados excluyendo los juegos del propio usuario
        recommended_games_indices = indices.flatten()[1:] # Excluir al propio usuario
        recommended_games = user_item_matrix.iloc[recommended_games_indices, :].mean(axis=0).sort_values(ascending=False)
        top_5_recommendations = recommended_games.index[:5]

        # Obtener los nombres de los juegos recomendados y sus géneros
        games_list = []
        for game_id in top_5_recommendations:
            game_data = df_model[df_model['item_id'] == game_id]
            if not game_data.empty:
                game_title = game_data['item_name'].iloc[0]
                game_genre = game_data['genres'].iloc[0]
                
                # Manejo de valores NaN
                if pd.isna(game_genre):
                    game_genre = "No especificado"

                games_list.append({'Titulo': game_title, 'Genero': game_genre})

        # Filtrar valores NaN de la lista de juegos
        games_list = [game for game in games_list if not any(pd.isna(val) for val in game.values())]

        return games_list[:5]  # Devolver la lista de juegos recomendados

    except ValueError as e:
        # Si el ID de usuario no es un string, devuelve un mensaje de error
        return {'Error': str(e)}