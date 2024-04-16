# <h1 align=center> <strong> **PROYECTO INDIVIDUAL Nº1** </strong> </h1>

# <h2 align="center"> **Machine Learning Operations (MLOps)** </h2>

<p align="center">
<img src='src\ML-dev-DE.png' height=300>
</p>

## **`Índice`**
- [Índice](#índice)
- [Introducción ](#introducción-)
- [Contenido del repositorio ](#contenido-del-repositorio-)
- [Información de los datos ](#información-de-los-datos-)
- [Propuesta de trabajo ](#propuesta-de-trabajo-)
- [ETL - EDA ](#etl---eda-)
- [Resultado ](#resultado-)
- [Herramientas ](#herramientas-)
- [Elaborador ](#elaborador-)


## **`Introducción`** <a name="introduccion"></a>
Bienvenidos al proyecto correspondiente a la etapa de Lab's de la carrera **Data Science** de **Henry**, donde nuestro rol asignado para esta ocasion es de **Data Engineer** en la empresa de Steam, una plataforma multinacional de videojuegos. 

Nuestro rol es crear un sistema de recomendación para los usuarios pero los datos entregados no estan listos para ser trabajados.



<p align="center">
  <img src='src\henry.png' height=300>
</p>

## **`Contenido del repositorio`** <a name="contenido"></a>
En este repositorio encontraran una carpeta y seis archivos:

* Encontraran los dataset siendo dos archivos:
> * *new_steam_games* : Datos limpios que se obtuvieron luego del proceso del ETL, con datos desanidados, preparacion de analisis de sentimiento y correción de tipográficos.
> * *modelo_de_prediccion* : csv con los datos de interés en las que se realizó el modelo de aprendizaje.
* Encontraran la carpeta **src** donde estan las imagenes utilizadas en el presente REAME.
* En archivo **ETL** encontraran toda la documentación y el paso a paso de lo que se trabajo hasta llegar a un CSV con datos limpios.
* En archivo **EDA** encontrara toda la documentación y el paso a paso de lo analizado para trabajar en la API.
* En archivo **main** donde encontraran el programa en el que contiene todas las funciones realizadas y su conexión con la API.
* En archivo **requirements** se encuentran las librerias utilizadas para que la API funcione.


## **`Información de los datos`** <a name="informacion"></a>
Los datos entregados para trabajar fueron tres y en un formato JSON
* "australian_user_items.json"
* "australian_user_reviews.json"
* "output_stream_games.json"

- Datasets entegados: [Enlace de los datasets](https://drive.google.com/drive/folders/1vHwfk7OJ_vb8Ar3DUzuyW6vUmpLGSMUq?usp=drive_link)

## **`Propuesta de trabajo`** <a name="propuesta"></a>

<p align="center">
  <img src='src\Procesos.png' height=300>
</p>

## **`ETL - EDA`** <a name="etl-eda"></a>

- Desanidar datos.
- Revisión, manejo y eliminación de nulos.
- Transformaciones de datos en columnas.
- Creación de nuevas columnas con datos relevantes y normalizados.
- Eliminación de columnas innecesarias.
- Eliminación de datos poco útiles.
- Exportación de los datos transformados a un nuevo CSV.

## **`Resultado`** <a name="resultado"></a>

<p align="center">
  <img src='src\API.jpeg' height=300>
</p>

El resultado es una API renderizada con las seis funciones requeridas.

- Acceder al siguiente link: [API]()
- Acceder a los CSV: [Datasets](https://drive.google.com/drive/folders/1Y7QCXQIjiI6eD7Gh7VLCbUlMdZ5cjhvi?usp=drive_link)

## **`Herramientas`** <a name="herramientas"></a>

- Python
- Pandas
- Numpy
- Fastapi
- Sklearn

## **`Elaborador`** <a name="elaborador"></a>

* Brenda Romero

>> Linkedin : https://www.linkedin.com/in/brenda-romerok/