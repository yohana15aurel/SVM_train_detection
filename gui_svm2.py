import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from time import time
import cv2
from PIL import ImageTk, Image
import numpy
from skimage.feature import hog

#load the trained model to classify the images

from tensorflow.keras.models import load_model
# model = load_model('model_ex-001_acc-0.250000.h5')
with open('finalized_model_dataset3.sav', 'rb') as file:
    model = pickle.load(file)

#dictionary to label all the CIFAR-10 dataset classes.

classes = {
    0:'accident',
    1:'dense_traffic',
    2:'fire',
    3:'sparse_traffic'
}

# classes=(['acc','dense_traffic', 'fire','sparse_traffic'])


# classes=['accident','dense_traffic', 'fire','sparse_traffic']
#initialise GUI

top=tk.Tk()
top.geometry('800x600')
top.title('Image Traffic Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))

sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    t2 = time()
    image = image.resize((128,128))
    image = cv2.cvtColor(numpy.float32(image), cv2.COLOR_BGR2GRAY)
    # image = numpy.expand_dims(image, axis=0)
    image, hog_data = hog(image, orientations=9, pixels_per_cell=(32, 32),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2', visualize=True)
    # image = image.reshape((1*128*128))
    image = numpy.array(image)
    pred = model.predict([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign)
    end = (time() - t2)
    print('prediction done in %0.3fs' % end)
    label_time.configure(foreground='#011638', text=end)

def processing(file_path):
    global label_packed
    image = Image.open(file_path)
    image.thumbnail(((top.winfo_width() / 2.25),
                        (top.winfo_height() / 2.25)))

    t2 = time()
    image = image.resize((128,128))
    image = cv2.cvtColor(numpy.float32(image), cv2.COLOR_BGR2GRAY)
    image = numpy.array(image)
    im = ImageTk.PhotoImage(image)
    sign_image.configure(image=im)
    sign_image.image = im
    label.configure(text='')


def show_classify_button(file_path):
    classify_b=Button(top,text="Classify",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def show_processing_button(file_path):
    processing_b=Button(top,text="processing",
   command=lambda: processing(file_path),padx=10,pady=5)
    processing_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    processing_b.place(relx=0.79,rely=0.56)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        # show_processing_button(file_path)
        show_classify_button(file_path)
    except:
        pass

def camera_open():
    with open('finalized_model_dataset3.sav', 'rb') as file:
        model = pickle.load(file)

    video = cv2.VideoCapture(0)
    try:
        while True:
            _, frame = video.read()

            # Convert the captured frame into RGB
            im = Image.fromarray(frame, 'RGB')

            # Resizing into 128x128 because we trained the model with this image size.
            im = im.resize((128, 128))
            im = cv2.cvtColor(numpy.float32(im), cv2.COLOR_BGR2GRAY)
            # image = np.expand_dims(im, axis=0)
            im, hog_data = hog(im, orientations=9, pixels_per_cell=(32, 32),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2', visualize=True)
            img_array = numpy.array(im)

            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            img_array = numpy.expand_dims(img_array, axis=0)

            # Calling the predict method on model to predict 'me' on the image
            prediction = int((model.predict(img_array)[0]))

            sign = classes[prediction]
            print(sign)

            # if prediction is 0, which means I am missing on the image, then show the frame in gray color.
            if prediction == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'accident', (155, 280), font, 2, (255, 0, 0), 1)
            if prediction == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'dense traffic', (55, 280), font, 2, (0, 0, 255), 1)
            if prediction == 2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'fire', (55, 280), font, 2, (0, 0, 255), 1)
            if prediction == 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'sparse traffic', (55, 280), font, 2, (0, 0, 255), 1)

            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
    except:
        pass

upload=Button(top,text="Upload Image",command=upload_image,
  padx=10,pady=5)

camera=Button(top,text="camera Image",command=camera_open,
  padx=10,pady=5)

upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

camera.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

camera.pack(side=BOTTOM,pady=6)

upload.pack(side=BOTTOM,pady=3)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="Traffic Classification",pady=20, font=('arial',20,'bold'))
label_time=Label(top,text='prediksi selesai dalam waktu', font=('arial',12,'bold'))

label_time.configure(background='#CDCDCD')
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
label_time.pack()
top.mainloop()

# d = {'John':5, 'Alex':10, 'Richard': 7}
# list = []
# for i in d:
#    k = (i,d[i])
#    list.append(k)
#    d.values()
#
# print (list)
# print(d.values())