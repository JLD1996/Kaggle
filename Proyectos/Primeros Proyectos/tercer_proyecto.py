#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:22:06 2022

@author: javier
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTANDO DATOS (df = dataframe = marco de datos)
house_df = pd.read_csv("precios_hogares.csv")

# VISUALIZACION
sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)

# CORRELACION
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(house_df.corr(), annot = True)

# LIMPIEZA DE DATOS
# Primero vamos a seleccionar cuáles van a ser las características de la casa con las cuales vamos a hacer
# la predicción. Entonces, en general, cualquier agente de bienes raíces sabrá qué criterios usar para ver los
# bienes de una casa. Pero, en este caso vamos a usar los siguientes: 
# bedrooms, bathrooms,  sqft_living (pies cuadrados de la casa), sqft_lot (cuantos pisos cuadrados tiene casa)
# Tenemos que coger aquellos datos que tengan mayor correlación en nuestro heatmap

selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

# Creamos nuestra matrix X. Marco de datos: 
# Valor de y, que es el que queremos predecir. 

x = house_df[selected_features]
y = house_df['price']    

# Si nos fijamos en x, tiene varias escalas. Entonces, lo escalamos con escalado de datos. 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# NORMALIZANDO OUTPUT

y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

# ENTRENAMIENTO
# Con la siguiente función crearemos nuestro set de entrenamiento
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.25)

#DEFINICION MODELO
model = tf.keras.Sequential()  
# Capa de entrada
model.add(tf.keras.layers.Dense(units=100, activation = 'relu', input_shape=[7,]))
# Capas ocultas
model.add(tf.keras.layers.Dense(units=100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units=100, activation = 'relu'))
# Capa de salida
model.add(tf.keras.layers.Dense(units=1, activation = 'linear'))

model.summary()

model.compile(optimizer='Adam', loss='mean_squared_error')
# batch_size = tamaño del lote
epochs_hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

# EVALUANDO MODELO
epochs_hist.history.keys()

# GRAFICO
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso de Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and validation loss')
plt.legend(['Training loss', 'Validation loss'])

# PREDICCION
# Definimos hogar para predecir con sus respectivas inputs
x_test_1 = np.array([[4,3,1960,5000,1,2000,3000]])

# Escalamos
scaler_1 = MinMaxScaler()
x_test_scaled_1 = scaler_1.fit_transform(x_test_1)

y_predict = model.predict(x_test_scaled_1)

# Como no esta en la escala que queremos, revertimos el escalado. 
y_predict = scaler.inverse_transform(y_predict)


