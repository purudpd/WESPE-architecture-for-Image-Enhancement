from __future__ import print_function
from scipy import misc
import os
import numpy as np
import sys
from glob import glob
import cv2
from random import shuffle

def load_batch(dped_dir, IMAGE_SIZE, LIMIT):

    print("Starting train load start!!")

    train_images_phone = glob(dped_dir + '\\sony\\*')
    train_images_Dslr = glob(dped_dir + '\\canon\\*')

    #To enforce mismatch
    c = list(zip(train_images_phone, train_images_Dslr))
    shuffle(c)
    train_images_phone, train_images_Dslr = zip(*c)

    train_images_phone = train_images_phone[:LIMIT] 
    train_images_Dslr = train_images_Dslr[:LIMIT] 

    train_data_phone = np.zeros((len(train_images_phone), IMAGE_SIZE))
    train_data_Dslr = np.zeros((len(train_images_Dslr), IMAGE_SIZE))

    for i,image_name in enumerate(train_images_phone):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.reshape(image, [1, IMAGE_SIZE]) / 255.0
        #image = 2.*(image - np.min(image))/np.ptp(image)-1
        train_data_phone[i,:] = image

    for i,image_name in enumerate(train_images_Dslr):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = np.reshape(image, [1, IMAGE_SIZE]) / 255.0
        #image = 2.*(image - np.min(image))/np.ptp(image)-1
        train_data_Dslr[i,:] = image

    return train_data_phone, train_data_Dslr