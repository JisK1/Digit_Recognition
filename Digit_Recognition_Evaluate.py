#Digit image recognition 
#Learned from this video "https://www.youtube.com/watch?v=bte8Er0QhDg" NeuralNine

#This script evaluates the model using the test data set.
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

#Load the saved model.
model = tf.keras.models.load_model('handwritten.model')

#evaluate based on the testdata
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)