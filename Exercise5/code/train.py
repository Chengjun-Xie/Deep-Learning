import torch as t
import torchvision as tv
from data import get_train_dataset, get_validation_dataset, ChallengeDataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import resnet

# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
#TODO： challenge point -> Batchsize
train_dl = t.utils.data.DataLoader(get_train_dataset(), batch_size=20)
val_dl = t.utils.data.DataLoader(get_validation_dataset(), batch_size=1)
data = ChallengeDataset()
# set up your model
model = resnet.ResNet18()

# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
# set up optimizer (see t.optim); 
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
#TODO：challenge point -> Hyperparameters
criterion = t.nn.BCEWithLogitsLoss(pos_weight=data.pos_weight())
optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
early_stop = EarlyStoppingCallback(10)

trainer_obj = Trainer(model, criterion, optimizer, train_dl, val_dl, cuda=False, early_stopping_cb=early_stop)
# go, go, go... call fit on trainer
res = []
res[0], res[1] = trainer_obj.fit(1)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')