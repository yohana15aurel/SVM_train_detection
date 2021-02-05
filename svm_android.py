import pickle
import sys
from tkinter import font

from skimage.feature import hog

import cv2
import numpy as np
from PIL import Image
from keras import models

# Load the saved model
# model = models.load_model('model_ex-001_acc-0.250000.h5')
with open('finalized_model_dataset3.sav', 'rb') as file:
    model = pickle.load(file)
classes = {
    0: 'accident',
    1: 'dense_traffic',
    2: 'fire',
    3: 'sparse_traffic'
}

class MobileCamera:
    def getVideo(self,camera):
        self.camera = camera
        video = cv2.VideoCapture(self.camera)
# video = cv2.VideoCapture(0)

        while True:
            _, frame = video.read()

            frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            # Convert the captured frame into RGB
            im = Image.fromarray(frame, 'RGB')


            # Resizing into 128x128 because we trained the model with this image size.
            im = im.resize((128, 128))
            im = cv2.cvtColor(np.float32(im), cv2.COLOR_BGR2GRAY)
            # image = numpy.expand_dims(image, axis=0)
            im, hog_data = hog(im, orientations=9, pixels_per_cell=(32, 32),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2', visualize=True)
            img_array = np.array(im)

            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            img_array = np.expand_dims(img_array, axis=0)

            # Calling the predict method on model to predict 'me' on the image
            prediction = int((model.predict(img_array)[0]))

            sign = classes[prediction]
            print(sign)

            # if prediction is 0, which means I am missing on the image, then show the frame in gray color.
            # if prediction == 0:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(frame, 'cannot Detect', (55, 280), font, 0.5, (0, 255, 0))
            if prediction == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'accident', (55, 280), font, 1, (0, 255, 0),2)
            if prediction == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'dense traffic', (55, 280), font, 1, (0, 255, 0),2)
            if prediction == 2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'fire', (55, 280), font, 1, (0, 255, 0),2)
            if prediction == 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'sparse traffic', (55, 280), font, 1, (0, 255, 0),2)

            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
video = MobileCamera()
video.getVideo("http://192.168.43.1:8080/video")