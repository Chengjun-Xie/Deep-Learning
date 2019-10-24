import os.path
import json
import random
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.iterationTime = 0
        self.randomNumberList = random.sample(range(0, 100), 100)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

        batch = []
        labels = []
        with open(self.label_path) as f:
            json_data = json.load(f)

        # enroll a dictionary to a list
        key = []
        value = []
        for k, v in json_data.items():
            key.append(k)
            value.append(v)

        for i in range(0, self.batch_size):
            n = i + self.iterationTime * self.batch_size
            if n >= 100:
                n -= 100

            # load path and label from .json file
            if self.shuffle:
                temp = int(self.randomNumberList[n])
                img_path = self.file_path + "/" + key[temp] + ".npy"
                labels.append(value[self.randomNumberList[n]])
            else:
                img_path = self.file_path + "/" + key[n] + ".npy"
                labels.append(value[n])

            # load image to batch
            img = np.load(img_path)
            img = scipy.misc.imresize(img, self.image_size)
            img = self.augment(img)
            batch.append(img)

        self.iterationTime += 1
        batch = np.array(batch)
        labels = np.array(labels)
        return batch, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        # rotation
        if self.rotation:
            k = np.random.randint(3) + 1
            img = np.rot90(img, k)
        # mirroring
        if self.mirroring:
            k = np.random.randint(1)
            img = np.flip(img,k)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method

        images, label = self.next()
        plt.figure('Image Generator')
        col = 4
        row = int(self.batch_size / col) + 1
        for i in range(0, self.batch_size):
            name = self.class_name(label[i])
            plt.subplot(col, row, i + 1)
            img = images[i]
            plt.imshow(img)
            plt.title(name)
        plt.show()


def main():
    npy_path = 'exercise_data'
    j_path = 'Labels.json'
    batch = 20
    size = (100, 100)
    test = ImageGenerator(npy_path, j_path, batch, size)
    test.show()


if __name__ == "__main__":
    main()
