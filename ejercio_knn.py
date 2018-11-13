import numpy as np #Importamos la utileria de Numpy
import pandas as pd #Importamos la utileria de pandas
import matplotlib #Importamos la matplotlib para poder graficar
import math #Importamos math para poder hacer calculos con potencias y raices
from copy import deepcopy
from matplotlib import pyplot as plt

#Funcion que calcula la Distancia Euclidiana
def dist(a, b):
    d = np.zeros(len(b))
    for i in range(len(b)):
        d[i] = math.sqrt(np.power(a[0] - b[i,0],2) + np.power(a[1] - b[i,1],2))
    return (d)
#Leemos los valores de un archivo para hacerlo dinamicos los datos
datos = pd.read_csv(r'''C:\Users\CGG\Documents\Diplomado\datos.csv''')
print("Tamano del Data Frame ", datos.shape)
#Cordenas de los puntos
x = datos['x'].values
y = datos['y'].values
#Guardamos los valores en un arreglo de numpy
x1 = np.array(list(zip(x, y)))
print ("Datos a Clasificar")
print(x1)
#Leemos de un archivo el No de clusters para hacer dinamico el valor
k_data = pd.read_csv(r'''C:\Users\CGG\Documents\Diplomado\cluster.csv''')
k = int(k_data['no_cluster'].values)
print("Tamano del Cluster y No de Repeticiones ")
repeticiones = int(k_data['no_rep'].values)
print(k,repeticiones)
print(" ")
#Obtnenemos las coordenas de los centros de manera aleatoria
print("Coordenadas Iniciales de los Centros")
C_x = np.random.randint(1, max(x), size=k)
C_y = np.random.randint(1, max(y), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.int)
print(C)

#Variables donde almacenameros los centros y lo clusteres generados
C_anterior = np.zeros(C.shape)
clusters = np.zeros(len(x1))
#Aplicamos algortimo de KNN para obtener los clusters y los centros para 20 repeticiones
print("Nuevos Centros")
rep = 0
while rep < repeticiones:
    for i in range(len(x1)):
        distancia = dist(x1[i], C) #Calcula la distancia euclidiana
        cluster = np.argmin(distancia) # Asignamos los valores a los clusters
        clusters[i] = cluster
    C_anterior = deepcopy(C) #Copias el centro
    # Calculos de los nuevos centros obtniendo el promedio
    for i in range(k):
        datos = [x1[j] for j in range(len(x1)) if clusters[j] == i]
        C[i] = np.mean(datos, axis=0)
    print("Centro ", rep)
    print (C)
    rep += 1

#Graficamos los clusters y los centros
plt.rcParams['figure.figsize'] = (6, 4)
plt.style.use('ggplot')
colors = ['b', 'm', 'r', 'y', 'c', 'g']
fig, ax = plt.subplots()
for i in range(k):
        datos = np.array([x1[j] for j in range(len(x1)) if clusters[j] == i])
        ax.scatter(datos[:, 0], datos[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()
