# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 00:39:45 2022

@author: HP
"""


import tensorflow as tf
import random
import numpy as np
import cv2

 
from keras.utils import normalize

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import model
import imageProcessing

patch_size=128


print(tf.__version__)
   #Modelcheckpoint
def check_point(modelFile):

       callbacks = [
               tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
               tf.keras.callbacks.TensorBoard(log_dir='logs'),
               tf.keras.callbacks.ModelCheckpoint(modelFile, verbose=1, save_best_only=True)
               ]


       return callbacks

def training():
    return
def predit_the_test_image(model1,X_train,X_test):
    preds_train = model1.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model1.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test = model1.predict(X_test, verbose=1)

     
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    
    return preds_train_t, preds_val_t, preds_test_t

def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 128):   #Steps of 128
        for j in range(0, image.shape[1], 128):  #Steps of 128
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img

if __name__ == "__main__":
    seed = 42
    np.random.seed = seed


    #Build the model
    model1 = model.build_unet()


    ################################
 

    X_train, Y_train,train_gen = imageProcessing.image_pre_processing_for_training()


    
    results = model.fit_generator(train_gen, validation_split=0.1, batch_size=16,
                         epochs=25, callbacks=check_point('model_for_nuclei.h5'))



    X_test, y_test = imageProcessing.image_pre_processing_for_testing()
    ####################################

    idx = random.randint(0, len(X_train))

    """
    preds_train = model1.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model1.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test = model1.predict(X_test, verbose=1)

     
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    
    """
    preds_train_t, preds_val_t, preds_test_t = predit_the_test_image(model1,X_train,X_test)
    
    
    # Perform a sanity check on some random training samples
    # validation and testing
    
    
    ix = random.randint(0, len(preds_train_t))
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0]*0.9):][ix])
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()
    
    #image
    image = cv2.imread('E:/HRF/HRF-all/images/08_g.jpg', 0)
    segmented_image = prediction(model, image, patch_size)
    plt.hist(segmented_image.flatten())  #Threshold everything above 0
    
    #ROC
    from sklearn.metrics import roc_curve
    y_preds = model.predict(X_test).ravel()

    fpr, tpr, thresholds = roc_curve(y_test, y_preds)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve on DRIVE')
    plt.show()

