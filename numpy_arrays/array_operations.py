import numpy as np

# Crea un array de 5x5 con valores aleatorios entre 0 y 100
array_aleatorio = np.random.randint(0, 101, size=(5, 5))
print("Array aleatorio:")
print(array_aleatorio)

# Suma de todos los elementos
suma_total = np.sum(array_aleatorio)
print("Suma total:", suma_total)

# Promedio de cada fila
promedio_filas = np.mean(array_aleatorio, axis=1)
print("Promedio de cada fila:", promedio_filas)
# Promedio de cada columna
promedio_columnas = np.mean(array_aleatorio, axis=0)
print("Promedio de cada columna:", promedio_columnas)

# Valor maximo
valor_maximo = np.max(array_aleatorio)
print("Valor maximo:", valor_maximo)
# Valor minimo
valor_minimo = np.min(array_aleatorio)
print("Valor minimo:", valor_minimo)