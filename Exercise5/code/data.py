from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
VAL_size = 0.2


class ChallengeDataset(Dataset):
    def __init__(self, phase="train", path="train.csv", val_size=0.2,
                 compose=tv.transforms.Compose([tv.transforms.ToTensor()])):
        """
        inherits from the torch.utils.data.Dataset class and provides some basic functionalities
        :param phase:   A mode ﬂag of type String which can be either “val” or “train”,
        :param path:    the path to the csv ﬁle
        :param val_size:   a parameter which controls the split between your train and validation data
        :param compose: The Compose object allows to easily perform a chain of transformations on the data
                        Among other aspects, this is interesting for data augmentation.
                        In the transpose package, you can ﬁnd diﬀerent augmentation strategies.
        """
        self.phase = phase
        self.path = path
        self.val_size = val_size
        self.compose = compose

        # split the dataset
        self.hole_data = pd.read_csv(self.path, sep=";")
        train_data, val_data = train_test_split(self.hole_data,
                                                test_size=self.val_size,
                                                random_state=1)
        if self.phase == "train":
            self.csv_data = train_data
        elif self.phase == "val":
            self.csv_data = val_data

    def __getitem__(self, item):
        """
        Before returning the sample, perform the transformations speciﬁed in the transform member.
        The two return values need to be of type [torch.tensor].
        :param item: index
        :return: returns the sample as a tuple: the image and the corresponding label
        """
        data = self.csv_data.iloc[item]
        image = imread(data.filename)
        image = gray2rgb(image)
        image = self.compose(image)

        label = np.zeros(2)
        label[0] = data.crack
        label[1] = data.inactive
        label = torch.from_numpy(label)
        return image, label

    def __len__(self):
        """
        :return: It returns the length of the currently loaded data split.
        """
        return len(self.csv_data.index)

    def pos_weight(self):
        """
        It should calculate a weight for positive examples for each class
        :return: return it as a [torch.tensor]
        """
        pos_weight = np.ones(2)
        pos_weight[0] = (1 - np.array(self.hole_data.crack)).sum()
        pos_weight[0] /= np.array(self.hole_data.crack).sum()
        pos_weight[0] = (1 - np.array(self.hole_data.inactive)).sum()
        pos_weight[0] /= np.array(self.hole_data.inactive).sum()
        return torch.from_numpy(pos_weight)


def get_train_dataset():
    # TODO：challenge point -> data augmentation
    tf_obj = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(mean=train_mean, std=train_std)])
    return ChallengeDataset(phase="train", val_size=VAL_size, compose=tf_obj)


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    tf_obj = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                    tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=train_mean, std=train_std)])
    return ChallengeDataset(phase="val", val_size=VAL_size, compose=tf_obj)
