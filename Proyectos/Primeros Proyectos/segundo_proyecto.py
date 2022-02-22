#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:12:42 2022

@author: javier
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTANDO DATOS (df = dataframe)
sales_df = pd.read_csv("datos_ventas.csv")

# VISUALIZACION
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

# CARGANDO SETS DE DATOS
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

# CREANDO EL MODELO

# Con keras armamos nuestro modelo. Esto hace que podamos crear nuestra red neuronal de una manera muy simple. 
# Sequential significa que vamos a crear nuestro modelo de forma secuencial, capa por capa.

model = tf.keras.Sequential()  

# Agregamos las primeras capas. La primera es la de input, que se refiere a cómo vamos a recibir la información
# en nuestra red neuronal. 'Dense' son las capas más simples que hay. Simplemente estamos creando una capa neu-
# ronal y podemos hiperconectarlas con más capas 'Dense' si queremos. En nuestro caso, solo haremos uso de una.
# Ponemos entonces que nuestra unidad de entrada 'units' es 1. 

model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Algo muy importante cuando trabajamos con redes neuronales más complejas es ver con que el que 
# estamos trabajando. Con el siguiente método vemos todas las capas de nuestro modelo. 

# model.summary()

# Las redes neuronales aprenden a través de la función de pérdida, y también tienen optimizadores que permiten 
# que la red funcione. Vamos a obtener todo esto a través del compilado. 

# Ponemos un optimizador que nos va a ayudar a crear las funciones de pérdida y optimizar el sesgo del modelo
# Con 'loss' declaramos la función de pérdida

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# ENTRENANDO EL MODELO
# Tenemos que crear una función que nos permita poder actualizar nuestros pesos y correr a través de nuestro set
# set de datos para que el modelo pueda entrenar. Un 'ephoc' significa que nuestra red ha pasado por todo el
# set de entrenamiento un total de 100 veces. 

epochs_hist = model.fit(x_train, y_train, epochs = 10000)

# Podemos ver que cada en cada epoch el modelo tiene menos 'loss', es decir, es más preciso. 
    

# EVALUACION DEL MODELO
# Esto la hacemos para ver como fue entrenando y vamos a ver la curva de entrenamiento del modelo. Vamos a 
# graficar nuestro modelo. Al igualar keys al resto, tenemos como parámetro a nuestra pérdida. 

keys = epochs_hist.history.keys()

# GRAFICO
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de pérdida durante entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend('Training loss')

# Los pesos es lo que hace que el modelo aprenda. Vamos a obtenerlos.

weights = model.get_weights()
print(weights)

# PREDICCIONES

# Vamos a hacer una predicción con nuestro modelo, y posteriormente con la fórmula. Con esto vemos cuán 
# eficiente es. 

temp = 50
# Toda la información de nuestra red neuronal se encuentra en 'model'. Al 'predict()' le indicamos que
# queremos predecir. Tenemos que pasarle un input, en este caso, la temperatura. El output será el revenue.
revenue = model.predict([temp])
print('La ganancia según la red neuronal será:', revenue)

# Por último mostraremos la correlación de nuestros precios y nuestra red.
plt.scatter(x_train, y_train, color='gray')
# y_train es lo que queremos predecir. Entonces pasamos lo siguiente:
plt.plot(x_train, model.predict([x_train]), color='red')
plt.xlabel('Temperatura ºC')
plt.ylabel('Ganancia en dólares')
plt.title('Ganancia generada vs Temperatura en ambiente')
