# Simple CNN implementation

import os

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

train_gen = None
test_gen = None
val_gen = None

def load_data(main_dir):
    train_dir = os.path.join(main_dir, 'train')
    test_dir = os.path.join(main_dir, 'test')
    val_dir = os.path.join(main_dir, 'val')

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')
    test_gen = test_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')

    print(train_gen)

if __name__ == '__main__':
    load_data('..\\dataset')