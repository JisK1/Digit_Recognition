#Digit image recognition 
#Learned from this video "https://www.youtube.com/watch?v=bte8Er0QhDg" NeuralNine

#This script uses a trained model to predict what digit is in an image.
from colorsys import yiq_to_rgb
import os
from pyexpat import model
from statistics import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load the model
model = tf.keras.models.load_model('handwritten.model')

#While loop which continues as long as there is a file in the digits directory of "digit{image_number}.png"
image_Number = 1
while os.path.isfile(f"digits/digit{image_Number}.png"):
    try:
        #get the image from the directory
        img = cv2.imread(f"digits/digit{image_Number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        #use the model to predict what digit the image is.
        prediction = model.predict(img)
        #print the prediction
        print(f"This image is probably a {np.argmax(prediction)}")
        #Use matplotlib.pyplot to show the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("ERROR!")
    finally:
        image_Number += 1