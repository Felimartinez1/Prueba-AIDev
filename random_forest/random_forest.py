import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets

iris_dataset = datasets.load_iris()

iris = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
iris['target'] = iris_dataset.target
# Separación de características y etiquetas
X = iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo Random Forest con 10 árboles
modelo_RF = RandomForestClassifier(n_estimators=10, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
modelo_RF.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de datos de test
predicciones = modelo_RF.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, predicciones)

print(f"Accuracy: {precision:.2f}")

# Mostrar la matriz de confusión
matriz_confusion = confusion_matrix(y_test, predicciones)
print("\nConfusion matrix:")
print(matriz_confusion)
