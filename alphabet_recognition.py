from re import L
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=7500, train_size=2500, random_state= 9)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ", accuracy)

video = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = video.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = grayscale.shape

        #top left and bottom right points to draw the rectangle
        upper_left = (int(width/2 - 56), int(height/2 -56))
        bottom_right = (int(width/2 + 56), int(height/2 +56))
        
        cv2.rectangle(grayscale, upper_left, bottom_right, (0, 255, 0), 2)
        
        region_of_interest = grayscale[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        pillow_image = Image.fromarray(region_of_interest)

        grayscale_image = pillow_image.convert("L")
        resized_image = grayscale_image.resize((22, 30), Image.ANTIALIAS)
        inverted_image = PIL.ImageOps.invert(resized_image)

        pixel_filter = 20

        min_pixel = np.percentile(inverted_image, pixel_filter)
        scaled_image = np.clip(inverted_image-min_pixel, 0, 255)
        max_pixel = np.max(inverted_image)

        scaled_image = np.asarray(scaled_image)/max_pixel

        sample = np.array(scaled_image).reshape(1, 660)

        prediction = clf.predict(sample)
        print("The predicted alphabet is -->", prediction)

        cv2.imshow('Alphabet', grayscale)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
    
        if key & 0xFF == 27:
            break
    
    except Exception as e:
        pass

video.release()
cv2.destroyAllWindows()