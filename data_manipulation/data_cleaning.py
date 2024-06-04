import pandas as pd

wine_quality = pd.read_csv('data_manipulation\data\winequality-red.csv')

# Imprimir las primeras 5 filas del DataFrame
print(wine_quality.head())

# Calcular la media de la columna 'alcohol'
mean_alcohol = wine_quality['alcohol'].mean()
print(f"Media del contenido de alcohol: {mean_alcohol}")

# Filtrar las filas donde la calidad del vino es mayor o igual a 7
high_quality_wines = wine_quality[wine_quality['quality'] >= 7]
print(high_quality_wines.head())

# Guardar el DataFrame filtrado en un archivo JSON
high_quality_wines.to_json('data_manipulation\data\high_quality_wines.json', orient='records', lines=True)