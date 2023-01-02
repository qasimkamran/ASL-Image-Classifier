# Simple CNN implementation

import os

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

class DataGenerator:

    def __init__(self, main_dir):
        self.load_data(main_dir)

    train_gen = None
    test_gen = None
    val_gen = None

    def load_data(self, main_dir):
        train_dir = os.path.join(main_dir, 'train')
        test_dir = os.path.join(main_dir, 'test')
        val_dir = os.path.join(main_dir, 'val')

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        self.train_gen = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')
        self.test_gen = test_datagen.flow_from_directory(test_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')
        self.val_gen = val_datagen.flow_from_directory(val_dir, target_size=(100, 100), batch_size=20, class_mode='categorical')

    def display_batch(self):
        assert self.train_gen is not None, f'train_gen is set to None'
        for data_batch, label_batch in self.train_gen:
            print('Data batch shape:', data_batch.shape)
            print('Label batch shape:', label_batch.shape)
            break
        fig = plt.figure(figsize=(10, 10))
        columns = 5
        rows = 5
        for i in range(1, 19):
            img = data_batch[i]
            ax = fig.add_subplot(rows, columns, i)
            plt.subplots_adjust(hspace=0.9, wspace=0.01)
            label = np.where(label_batch[i] == 1)
            ax.title.set_text(" label = " + str(label[0]))
            plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    data_generator = DataGenerator('..\\dataset')
    data_generator.display_batch()