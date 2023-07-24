#Digit image recognition 
#Learned from this video "https://www.youtube.com/watch?v=bte8Er0QhDg" NeuralNine

#This script trains a model to recognize digits using tensorflow's mnist data set.
from colorsys import yiq_to_rgb
import os
from pyexpat import model
from statistics import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Get the data set
mnist = tf.keras.datasets.mnist
#split the data into testing and trainning data. This dataset already does it for us.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normilize and process the trainning and test data.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Create the model 
model = tf.keras.models.Sequential()
#Add input layer. This layer takes the 28x28 pixel image and turns it into a flat line of 784 pixles.
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#Basic Dense layer. Each neuron is connected to each other neuron of the other layers. activation function - rectified linear unit
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

#The output layer has 10 neurons for each of the 10 digits (0-9). the softmax function makes sure that all the 10 neurons add up to one. This lets us see the confidence in result for each digit.
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compile the model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model. epochs = number of interations.
model.fit(x_train, y_train, epochs=7)

model.save('handwritten.model')