import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# Generar datos normales
n_normal = 150
X_normal = np.random.randn(n_normal, 2)

# Generar datos de anomalías
n_anomaly = 40
radius = 7
center = np.random.rand(2)
X_anomaly = np.random.randn(n_anomaly, 2) * radius + center

# Combinar datos y agregar etiquetas
X = np.concatenate((X_normal, X_anomaly), axis=0)
y = np.concatenate((np.ones(n_normal), np.zeros(n_anomaly)), axis=0)

# Definir el modelo One-Class SVM
clf = OneClassSVM(kernel='rbf', nu=0.2)
# Entrenar el modelo con datos normales
clf.fit(X_normal, y=y[0:n_normal])
# Obtener la distancia a la frontera de decisión
decision_values = clf.decision_function(X)
# Clasificamos los puntos de datos como normales o anómalos según la distancia a la frontera
predictions = np.sign(decision_values)

# Vizz
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm_r')
plt.title('Detección de anomalías con One-Class SVM')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
