import pickle
import random
import time

#from datetime import datetime
from time import time
# # import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import skimage
from mlxtend.plotting import plot_decision_regions
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from skimage.exposure import exposure
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, precision_score, \
    f1_score, auc, roc_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm, metrics, ensemble


#dataset_path3 and dataset_path4 : 90% train size 10% test size (random)
#dataset_path and dataset_path2 : 80% train size and 20% test size (random)

trn_img_path = "dataset_path5/train"

# The testing data set is in the /Users/macos/Documents/Intel Image Classification/seg_test
tst_img_path = "dataset_path5/test"

# Lets create 2 set of arrays for train & testing data's. One for to store the Image data and anther one for label details
X_train =[] # Stores the training image hog data
label_train = [] # Stores the training image label

X_test = [] # Stores the testing image hog data
label_test = [] # Stores the testing image label

hog_images = []
hog_features = []

x_train = []
lbl_train = []

x_test = []


scene_label=['accident','dense_traffic', 'fire','sparse_traffic']

def hog_data_extractor(jpeg_path):
    # img_to_yuv = cv2.cvtColor(jpeg_data, cv2.COLOR_BGR2YUV)
    # img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
    # hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    #read the image from path
    jpeg_data = cv2.imread(jpeg_path)
    #resize image
    jpeg_data = cv2.resize(jpeg_data, (128, 128))
    #change resize image into grayscale
    jpeg_data_gray = cv2.cvtColor(jpeg_data, cv2.COLOR_BGR2GRAY)
    #input grayscale image into HOG and get the features
    fd,hog_data = hog(jpeg_data_gray, orientations=9, pixels_per_cell=(32,32),
                         cells_per_block=(2,2),transform_sqrt=True, block_norm='L2', visualize=True)

    #jpeg_data_gray = jpeg_data_gray.reshape(1*128*128)
    return fd

#load image training and testing
def jpeg_to_array (scene_type, img_root_path,data_type):
    scene_path = os.path.join(img_root_path,scene_type.lower())
    print('Loading ' + data_type +' untuk '+scene_type)
    for img in os.listdir(scene_path):
        img_path = os.path.join(scene_path,img)
        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            if(data_type == 'Training'):
                X_train.append(hog_data_extractor(img_path))
                label_train.append(str(scene_type))
            if(data_type =='Testing'):
                X_test.append(hog_data_extractor(img_path))
                x_test.append(img_path)
                label_test.append(np.array(str(scene_type)))

def load_img():
    [jpeg_to_array(scene,trn_img_path,'Training')for scene in scene_label]
    print("Jumlah data training : ",len(X_train))
    [jpeg_to_array(scene,tst_img_path,'Testing')for scene in scene_label]
    print("Jumlah data testing : ",len(X_test))


def train():
    le = LabelEncoder()
    y_train = le.fit_transform(label_train)
    y_test = le.fit_transform(label_test)

    t0 = time()

    # pca = PCA(n_components=7)  # here you can change this number to play around
    # x_train = pca.fit_transform(X_train)
    # x_test = pca.transform(X_test)

    # defining parameter range
    # param_grid = {'C': [1, 10, 2, 4, 8 ],
    #               'gamma': [1, 0.1, 10 ],
    #               'kernel': ['rbf']}

    # parameters = [
    #     {'C': [1,2,4,8,10], 'kernel': ['rbf'], 'gamma': [0.1, 1,2,4,10,0.125, 0.15, 0.17, 0.2]}]
    #
    #
    # #do a gridsearch to search the best parameters
    # clf = GridSearchCV(SVC(), parameters, refit=True, verbose=3,scoring='accuracy',
    #                           n_jobs=-1)

    # clf=OneVsRestClassifier(SVC(kernel='rbf',C=10,gamma=1))
    clf=svm.SVC(kernel='rbf', gamma=1, C=4)
    #clf = SVC(C=10, kernel='rbf', gamma=1, tol=0.00001)

    # fitting the model for grid search
    clf.fit(X_train, y_train)
    print("%d support vectors out of %d points" % (len(clf.support_vectors_), len(X_train)))
    # print('w = ', clf.coef_)
    print('b = ', clf.intercept_)
    print('Indices of support vectors = ', clf.support_)
    print('Support vectors = ', clf.support_vectors_)
    print('Number of support vectors for each class = ', clf.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))



    # grid_search = GridSearchCV(estimator=clf, param_grid=parameters, scoring='accuracy', cv=10,
    #                            n_jobs=-1)
    # grid_search = grid_search.fit(X_train, y_train)

    # best_accuracy = clf.best_score_
    # print('best accuracy',best_accuracy)

    # start_time = time.time()

    # print best parameter after tuning
    # print('best param',clf.best_params_)

    # print how our model looks after hyper-parameter tuning
    # print('best estimator',clf.best_estimator_)


    # print classification report
    # print(classification_report(y_test, grid_predictions))
    # print('accuracy',accuracy_score(y_test,grid_predictions))
    t1 = time()
    y_pred = clf.predict(X_test)

    print('classfication and prediction done in %0.3fs' % (time() - t1))
    accuracy = accuracy_score(y_test, y_pred)
    #print('Accureacy on training set: {:.2f}'.format(clf.score(X_train,y_train)))
    # print('Accureacy on test set: {:.2f}'.format(clf.score(X_test,y_test)))
    #accuracy
    print('Model accuracy is: ', accuracy)
    #F1 Score
    print('F1 score:', f1_score(y_test, y_pred,
                          average='weighted'))
    #Recall
    print('Recall:', recall_score(y_test, y_pred,
                            average='weighted'))
    #Precision
    print('Precision:', precision_score(y_test, y_pred,
                                  average='weighted'))
    # print("waktu yang dihabiskan : ", time.time() - start_time, "to run")

    #Show Classification report
    print(metrics.classification_report(y_test,y_pred))

    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')


    #Save the SVM model into sav file
    filename = 'finalized_model_dataset3.sav'
    pickle.dump(clf, open(filename, 'wb'))

    #Show the time that used for create model
    print('modelling done in %0.3fs' % (time() - t0))


def scene_predict(img_path):
    image = cv2.imread(img_path)
    ip_image = Image.open(img_path)
    image = cv2.resize(image, (128, 128))
    prd_image_data = hog_data_extractor(img_path)

    # Load model from file
    with open('finalized_model_dataset3.sav', 'rb') as file:
        pickle_model = pickle.load(file)

    scene_predicted = pickle_model.predict(prd_image_data.reshape(1, -1))[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(ip_image)
    ax[0].set_title('input image')

    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Image predicted :' + scene_label[scene_predicted]);

#     #print("waktu yang dihabiskan : ", time.time() - t2, "to run")



def predict():
    t2 = time()
    # start_time = time.time()
    # img_file = ['images_765.jpg']
    img_file = ['images_765.jpg','images_1.jpg','images_749.jpg','images_2.jpg','images_3.jpg','images_4.jpg','images_0081.jpg','images_018.jpg','images_708.jpg','images_008.jpg','images_012.jpg','images_026.jpg','images_027.jpg','images_712.jpg']
    for item in img_file:
        scene_predict('image/'+item)
    print('prediction done in %0.3fs' % (time() - t2))
    plt.show()
    # print("waktu yang dihabiskan : ", time.time() - start_time, "s")

#uncomment to run train model
load_img()
train()

#uncomment to run predict image
# predict()



