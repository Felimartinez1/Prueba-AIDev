import requests
from bs4 import BeautifulSoup
import json

url = 'https://peppersarg.com/salefinal/'
def extraer_producto(producto):
    """
    Extrae el titulo y precio del producto dado.

    Args:
        producto (elemento BeautifulSoup): Elemento HTML que representa un producto.

    Returns:
        dict: Diccionario con título y precio.
    """
    
    titulo = producto.find("div", class_="js-item-name item-name mb-2 font-small opacity-80").text.strip()
    precio = producto.find("span", class_="js-price-display item-price font-weight-bold").text.strip()

    return {
        "titulo": titulo,
        "precio": precio
        }
    
# Petición HTTP a la página
response = requests.get(url)
# Verificar si la petición fue exitosa
if response.status_code == 200:
    # Parsear HTML con BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
else:
    print("Error al acceder a la pagina:", response.status_code)
    exit()
# Encontrar elementos que contienen productos
productos = soup.find_all("div", class_="js-item-product col-6 col-md-3 item-product col-grid")[:10]  # Ajustar la clase según la estructura de la página

# Extraer datos de cada producto y guardarlo en una lista
datos_productos = []
for producto in productos:
    datos_producto = extraer_producto(producto)
    datos_productos.append(datos_producto)

# Guardar datos en archivo JSON
with open("web_scraping\data\datos_productos.json", "w") as archivo:
    json.dump(datos_productos, archivo, indent=4)

print("Datos extraidos y guardados en datos_productos.json")