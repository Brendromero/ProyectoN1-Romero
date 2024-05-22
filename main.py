from fastapi import FastAPI
import pandas as pd
import numpy as np
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


@app.get('/Desarrollador/', tags=['General'])
def developer(desarrollador: str):
    """Ingrese un desarrollador para obtener la cantidad de items y porcentaje de contenido Free por cada año"""
    # Verifico si el tipo de dato de 'desarrollador' es una cadena
    if not isinstance(desarrollador, str):
        return {'Error': 'Tipo de dato debe ser un string.'}

    try:
        # Convierto el nombre del desarrollador a minúsculas
        desarrollador = desarrollador.lower()
        
        # Tomo las columnas que se usarán del dataframe
        dataframe_reducido = df[['developer', 'release_date', 'item_id', 'price']]
        empresa_desarrolladora = dataframe_reducido[dataframe_reducido['developer'] == desarrollador]

        # Extraigo el año desde la columna release_date y se crea una nueva columna llamada 'year'
        empresa_desarrolladora['year'] = pd.to_datetime(empresa_desarrolladora['release_date']).dt.year

        # Creo una lista para almacenar los resultados
        result_list = []

        # Agrupo por año e iteramos por cada grupo para realizar operaciones
        for year, group in empresa_desarrolladora.groupby('year'):
            total_items = len(group)
            free_items = len(group[group['price'] == 'Free to Play'])
            free_percentage = (free_items / total_items) * 100 if total_items > 0 else 0
        
            result_list.append({
                'Year': year,
                'Total_Items': total_items,
                'Free_Percentage': free_percentage
            })

        # Retorno la lista de resultados si se encontraron resultados, de lo contrario, devuelve un mensaje de error
        if result_list:
            return result_list
        else:
            return {'Error': 'No se ha encontrado ningún desarrollador con ese nombre'}
    except Exception as e:
        # Capturo cualquier otra excepción y devuelve el mensaje de error
        return {'Error': str(e)}


@app.get('/Datos de Usuario/{id}', tags=['General'])
def userdata(user_id: str):
    """Ingrese el id de usuario para obtener la cantidad de dinero gastado por el mismo, el porcentaje de recomendación y cantidad de items."""
    try:
        # Convierto el nombre del desarrollador a minúsculas
        user_id = user_id.lower()
        
        if not isinstance(user_id, str):
            raise ValueError({'Error': 'Tipo de dato debe ser un string.'})
        
        usuario = df[['item_id', 'user_id', 'items_count', 'price']]

        # En df_price1 copiamos los resultados del DataFrame
        df_price1 = usuario.copy()

        # Convertimos 'price' a tipo string y eliminamos filas con valores específicos
        df_price1['price'] = df_price1['price'].astype(str)
        df_price1 = df_price1[~df_price1['price'].str.lower().isin(['free to play', 'free', 'nan'])]

        # Filtramos desde el user_id y fusionamos los resultados
        usuario_filtrado = usuario[usuario['user_id'] == user_id].copy()
        usuario_filtrado = pd.merge(usuario_filtrado, df_price1, on='item_id', how='left', suffixes=('_orig', '_new'))

        # Convertimos 'price' a tipo numérico, reemplazamos errores con NaN y luego eliminamos NaN
        usuario_filtrado['price'] = pd.to_numeric(usuario_filtrado['price_new'], errors='coerce')
        usuario_filtrado = usuario_filtrado.dropna(subset=['price'])

        dinero_gastado = usuario_filtrado['price'].sum().round(2)

        # Nuevamente para el porcentaje, filtramos por user_id
        porcentaje_recomendacion = df[df['user_id'] == user_id]
    
        # Con len se calcula la longuitud de porcentaje_recomendacion y lo guardamos en la variable
        reviews_totales = len(porcentaje_recomendacion)
    
        # Si no hay ningun registro, sera 0
        if reviews_totales == 0:
            porcentaje_total = 0
        # Si hay registros
        else:
            # Entonces se filtra los registros donde la columna de recommend es True y se contara la cantidad de registros
            reviews_positivos = len(porcentaje_recomendacion[porcentaje_recomendacion['recommend'] == True])
            # Calculamos el porcentaje de recomendaciones positivas como la proporcion de la linea anterior y reviews_totales
            # Y se multipicara por 100 para obtener un porcentaje
            porcentaje_total = (reviews_positivos / reviews_totales) * 100

            # Se toma con iloc el primer valor que la columna de items_count aparece para guardarlo en cantidad_items
            cantidad_items = usuario_filtrado['items_count_new'].iloc[0] if not usuario_filtrado.empty and 'items_count_new' in usuario_filtrado.columns else 0


            resultado = {
            'Usuario' : user_id,
            'Dinero gastado' : float(dinero_gastado),  # Convirtiendo a float de Python nativo
            '% de recomendación' : float(porcentaje_total),  # Convirtiendo a float de Python nativo
            'Cantidad de items' : int(cantidad_items)  # Convirtiendo a int de Python nativo
                    }

            return resultado
    except ValueError as e:
        # Si no es un string devuelve error:
        return str(e)
    else:
        return {'Error': 'No se ha encontrado ningun usuario con ese nombre'}



# Convierto las cadenas de la columna 'genres' en listas reales de Python utilizando ast.literal_eval()
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        return np.nan

df['genres'] = df['genres'].apply(safe_literal_eval)

# Utilizo la función explode() para convertir las listas de la columna 'genres' en filas separadas
df_expanded = df.explode('genres')

@app.get('/Usuario por genero/', tags=['General'])
def userforgenre(genero: str):
    """Ingrese el género para obtener el usuario que acumula mas horas de jugadas por dicho género y una lista de acumulación de horas jugadas por año de lanzamiento."""
    try:
        # Convertir el nombre del desarrollador a minúsculas
        genero = genero.lower()
        
        if not isinstance(genero, str):
            raise ValueError({'Error': 'Tipo de dato debe ser un string.'})
        
        # Se toma las columnas de los DataFrame y usamos el dropna para las columnas 'genres y 'pplaytime_forever'
        generos = df_expanded[['item_id', 'user_id', 'release_date', 'playtime_forever', 'genres']]
        generos = generos.dropna(subset=['genres', 'playtime_forever'])

        # Filtro por el género específico
        generos = generos[generos['genres'] == genero]

        # Verifico si el DataFrame filtrado esta vacio
        if generos.empty:
            return {'Error': 'No se encontraron registros para el género especificado.'}

        # Encontramos el usuario con más horas jugadas por el género
        max_hours_user = generos.loc[generos['playtime_forever'].idxmax()]

        # Asegurémonos de que 'playtime_forever' sea numérica
        generos['playtime_forever'] = pd.to_numeric(generos['playtime_forever'], errors='coerce')

        # Convierto la columna 'release_date' a datetime
        generos['release_date'] = pd.to_datetime(generos['release_date'], errors='coerce')

        # Extraigo el año en una columna separada
        generos['year'] = generos['release_date'].dt.year

        # Calculo la acumulación de horas jugadas por año de lanzamiento
        hours_by_year = generos.groupby('year')['playtime_forever'].sum().reset_index()

        # Ordeno el DataFrame por año
        hours_by_year = hours_by_year.sort_values(by='year')

        return {
            'Max_hours_user': {
                'Usuario': max_hours_user['user_id'],
                'Horas jugadas': hours_by_year['playtime_forever'].sum()
                    },
            'hours_by_year': hours_by_year.to_dict(orient='records')
                    }
    except ValueError as e:
        #Si no es un string devuelve error:
        return str(e)
    
@app.get('/Mejores desarrolladores por anio', tags=['General'])
def best_developer_year(anio: int):
    """Ingrese el año para obtener el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado."""
    try:
        if not isinstance(anio, int):
            raise ValueError({'Error': 'Tipo de dato debe ser un entero.'})
        
        # Se toma las columnas necesarias del DataFrame df
        usuario = df[['item_id','user_id','recommend', 'sentiment_analysis', 'release_date', 'developer']]
    
        # Filtro por el año específico
        usuario['release_date'] = pd.to_datetime(usuario['release_date'], errors='coerce')
        usuario = usuario[usuario['release_date'].dt.year == anio]

        # Verifico si el DataFrame filtrado esta vacio
        if usuario.empty:
            return {'Error': 'El año ingresado es incorrecto o no se han encontrado datos'}

        # Filtro por juegos recomendados y comentarios positivos
        juegos_recomendados = usuario[(usuario['recommend'] == True) & (usuario['sentiment_analysis'] == 2)]

        # Agrupo por desarrollador y contar la cantidad de juegos recomendados
        developer_counts = juegos_recomendados['developer'].value_counts().reset_index()
        developer_counts.columns = ['developer', 'item_id']

        # Ordeno por la cantidad de juegos recomendados en orden descendente
        developer_counts = developer_counts.sort_values(by='item_id', ascending=False).reset_index(drop=True)

        # Tomo los top 3 desarrolladores
        top_3_developers = developer_counts.head(3)

        return top_3_developers.to_dict(orient='records')

    except ValueError as e:
        # Si no es un entero devuelve error:
        return str(e)
    else:
        return {'Error': 'No se ha encontrado ese año'}

def get_review_counts(df):
    

    positive_reviews = len(df[df['sentiment_analysis'] == 2])
    negative_reviews = len(df[df['sentiment_analysis'] == 0])

    return {
        "Positive Reviews": positive_reviews,
        "Negative Reviews": negative_reviews
    }

@app.get('/Analisis de sentimiento/', tags=['General'])
def developer_reviews_analysis(desarrollador: str):
    """Ingrese el desarrollador para obtener un diccionario con el nombre del mismo y una lista con la cantidad total de registros de reseñas de usuarios."""
    try:
        # Convierto el nombre del desarrollador a minúsculas
        desarrollador = desarrollador.lower()
        
        # Filtro las reseñas para la desarrolladora proporcionada
        desarrolladora_reviews = df[df['developer'] == desarrollador].copy()
        
        # Verifico si el DataFrame filtrado esta vacio
        if desarrolladora_reviews.empty:
            return {'Error': 'No se ha encontrado ningun desarrollador con ese nombre'}

        # Utilizo la función get_review_counts para contar las reseñas positivas y negativas
        review_counts = get_review_counts(desarrolladora_reviews)

        # Devuelve el resultado en un diccionario
        result = {
            'Desarrolladora': desarrollador,
            'Reseñas Positivas': review_counts['Positive Reviews'],
            'Reseñas Negativas': review_counts['Negative Reviews']
                }
        # Retornamos el resultado
        return result
    except ValueError as e:
        # Si no es un string devuelve error:
        print(e)
        return {'Error': 'Ha ocurrido un error al procesar la solicitud'}

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