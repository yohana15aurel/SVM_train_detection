# import cv2
#
# from skimage.feature import hog
# from sklearn.svm import SVC
#
# cap = cv2.VideoCapture(0)
# dim = 128 # For HOG
#
# while True:
#     img = []
#     label = []
#     # Capture the frame
#     ret, frame = cap.read()
#
#     # Show the image on the screen
#     cv2.imshow('Webcam', frame)
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Convert the image into a HOG descriptor
#     gray = cv2.resize(gray, (dim, dim), interpolation = cv2.INTER_AREA)
#     #features = hog.compute(gray)
#     features= hog(gray, orientations=9, pixels_per_cell=(32,32),
#                          cells_per_block=(2,2),transform_sqrt=True, block_norm='L2', visualize=True)
#     # features = features.T # Transpose so that the feature is in a single row
#     clf = SVC(kernel='rbf', gamma=1, C=4)
#
#     clf = clf.fit(gray,features)
#
#     # Predict the label
#     pred = clf.predict(features)
#
#     # Show the label on the screen
#     print("The label of the image is: " + str(pred))
#
#     # Pause for 25 ms and keep going until you push q on the keyboard
#     if cv2.waitKey(25) == ord('q'):
#         break
#
# cap.release() # Release the camera resource
# cv2.destroyAllWindows() # Close the image window
import pickle
from tkinter import font

from skimage.feature import hog

import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved model
# model = models.load_model('model_ex-001_acc-0.250000.h5')
with open('finalized_model_dataset3.sav', 'rb') as file:
    model = pickle.load(file)

video = cv2.VideoCapture(0)
classes = {
    0:'accident',
    1:'dense_traffic',
    2:'fire',
    3:'sparse_traffic'
}
while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        im = cv2.cvtColor(np.float32(im), cv2.COLOR_BGR2GRAY)
        # image = np.expand_dims(im, axis=0)
        im, hog_data = hog(im, orientations=9, pixels_per_cell=(32, 32),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2', visualize=True)
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = int((model.predict(img_array)[0]))

        sign = classes[prediction]
        print(sign)

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'accident', (155, 280), font, 2, (255, 0, 0),1)
        if prediction == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'dense traffic', (55, 280), font, 2, (0, 0, 255),1)
        if prediction == 2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'fire', (55, 280), font, 2, (0, 0, 255),1)
        if prediction == 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'sparse traffic', (55, 280), font, 2, (0, 0, 255),1)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()