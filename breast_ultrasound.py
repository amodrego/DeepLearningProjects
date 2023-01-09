# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:43:21 2023

@author: alexm

RED NEURONAL BASADA EN IMAGENES DE ULTRASONIDOS DE MAMA. 
ENTRENADA CON IMAGENES DE TUMORES BENIGNOS, MALIGNOS Y SIN TUMOR

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

"""
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import pathlib


# =============================================================================
# Vectores que usaremos para almacenar las imagenes (x) 
# y las etiquetas de cada categoria (y) --> benign, malign y normal
# =============================================================================
x_data = []
y_data = []
# =============================================================================
# =============================================================================
# =============================================================================

tamano_imagen = 300


# Bucle para recuperar las imagenes de tumores benignos
input_images_dir = 'benign/'
files_names = os.listdir(input_images_dir)
for i, image_path in enumerate(files_names):
    
    
    image = cv2.imread(input_images_dir + image_path)
    image = cv2.resize(image, (tamano_imagen, tamano_imagen))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(tamano_imagen, tamano_imagen, 3)
    
    x_data.append(image)
    y_data.append(0)    
    
    
# =============================================================================
# Código para ver las 25 primeras imagenes del set de datos
# =============================================================================
# for i, image in enumerate(x_data):
#     if i < 25:
#         plt.subplot(5, 5, i+1)

#         # Para que los numeros de coordenadas no aparezcan en las imagenes cuando las mostramos
#         plt.xticks([])
#         plt.yticks([])
        
#         plt.imshow(image)
#     else:
#         break
        
    
    
# Bucle para recuperar las imagenes de tumores malignos
input_images_dir = 'malignant/'
files_names = os.listdir(input_images_dir)
for i, image_path in enumerate(files_names):
    
    
    image = cv2.imread(input_images_dir + image_path)
    image = cv2.resize(image, (tamano_imagen, tamano_imagen))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(tamano_imagen, tamano_imagen, 3)
    
    x_data.append(image)
    y_data.append(1) 



# Bucle para recuperar las imagenes sin tumor
input_images_dir = 'normal/'
files_names = os.listdir(input_images_dir)
for i, image_path in enumerate(files_names):
    
    
    image = cv2.imread(input_images_dir + image_path)
    image = cv2.resize(image, (tamano_imagen, tamano_imagen))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(tamano_imagen, tamano_imagen, 3)
    
    x_data.append(image)
    y_data.append(2) 
    


# =============================================================================
# Las 3 posibles situaciones se clasifican en numeros de 0-2 de forma que:
#    - 0 = Benign
#    - 1 = Malignant
#    - 2 = Normal
# =============================================================================


# =============================================================================
# Una vez tenemos los datos ya recuperados en dos listas distintas, se normalizan las
# imagenes y se convierten las listas de datos en arrays 
# =============================================================================
# Normalizar los datos del vector de imagenes y convertir las imagenes a array
x_data = np.array(x_data).astype(float) / 255
y_data = np.array(y_data)



# =============================================================================
# Montamos la Red Neuronal Convolucional. 
# Primero probamos sin aumento de datos y vemos como funciona
# =============================================================================
import tensorflow as tf
cnn_model = tf.keras.models.Sequential()

cnn_model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(300, 300, 3), activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn_model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn_model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn_model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn_model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn_model.add(tf.keras.layers.Flatten())

cnn_model.add(tf.keras.layers.Dense(100, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(3, activation='softmax'))


# =============================================================================
# Compilamos el modelo creado
# =============================================================================
cnn_model.compile(optimizer='adam',
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])


# =============================================================================
# Añadimos aumento de datos para poder tener una mayor cantidad de imagenes 
# con las que entrenar sin generar overfitting
# =============================================================================
# from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rotation_range=50, # El generador rotara las imagenes de manera aleatoria para aumentar el numero de imagenes disponibles
#                              width_shift_range=0.5, # Mover las imagenes en horizontal desde 0 hasta 1 
#                              height_shift_range=0.5, # Mover las imagenes en vertical
#                              shear_range=15, # Inclinar las imagenes
#                              zoom_range=[0.5, 1.5], # Permite hacer zoom en un rango a las imagenes
#                              vertical_flip=True, # Girar verticalmente las imagenes
#                              horizontal_flip=True, #  Girar horizontalmente las imagenes
                             
#                              )

# datagen.fit(x_data)

# Para poder ver unas cuantas imagenes del lote aumentado
# for imagen, etiqueta in datagen.flow(x_data, y_data, batch_size=10, shuffle=False):

    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(imagen[i], cmap='gray')
    
    # break


# Limpiamos algo de memoria antes de entrenar el modelo
import gc
gc.collect()

# =============================================================================
# Entrenamiento del modelo
# =============================================================================

# Separamos en train y test (si hacemos aumento de datos es obligatorio hacerlo)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.85, shuffle=True) 

# datagen_train = datagen.flow(x_train, y_train, batch_size=32)

cnn_model.fit(x_train, y_train, batch_size=32, epochs=10,
              validation_data=(x_test, y_test),
              
              steps_per_epoch= int(np.ceil(len(x_train) / 32)), # np.ceil() redondea el float que tenemos al int inmediatamente superior (redondeo a las unidades)
              validation_steps= int(np.ceil(len(x_test) / 32)),
              
              
              verbose=1
              
              )


