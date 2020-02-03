import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


# from evaluation import create_evaluation

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimiser
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_cb=None):  # The stopping criterion.
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        out = self._model(x)
        out = out.double()
        loss = self._crit(out, y)
        loss.backward()
        self._optim.step()
        return loss

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        out = self._model(x)
        out = out.double()
        loss = self._crit(out, y)
        return loss, out.round()

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()
        device = t.device("cuda" if self._cuda else "cpu")
        sum_loss = 0.0
        length = len(self._train_dl)

        for i, data in enumerate(tqdm(self._train_dl, desc='training'), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loss = self.train_step(inputs, labels)
            sum_loss += loss

        return sum_loss / length

    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice.
        # You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        device = t.device("cuda" if self._cuda else "cpu")
        score = []
        length = len(self._val_test_dl)

        with t.no_grad():
            self._model.eval()
            for i, data in enumerate(tqdm(self._val_test_dl, desc='testing'), 0):
                inputs, labels = data 
                inputs, labels = inputs.to(device), labels.to(device)
                loss, prediction = self.val_test_step(inputs, labels)

                f1_crack = f1_score(labels.cpu().numpy()[:, 0],
                                    prediction.cpu().numpy()[:, 0])
                f1_inactive = f1_score(labels.cpu().numpy()[:, 1],
                                       prediction.cpu().numpy()[:, 1])
                score.append(f1_crack)
                score.append(f1_inactive)

        loss = loss.sum() / length
        return loss, score

    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        loss = [[], []]
        score = []
        counter = 0

        self._model.zero_grad()
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists 
            # use the save_checkpoint function to save the model for each epoch
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # return the loss lists for both training and validation
            print("======= Epoch %.03d =======" % (counter + 1))
            if counter >= epochs:
                break

            t_loss = self.train_epoch()
            v_loss, score = self.val_test()
            loss[0].append(t_loss.cpu().detach().numpy())
            loss[1].append(v_loss.cpu().detach().numpy())

            self.save_checkpoint(counter)
            self._early_stopping_cb.step(t_loss)
            if self._early_stopping_cb.should_stop():
                break
            counter += 1
            print("training loss:", loss[0])
            print("validation loss:", loss[1])
            print("score crack:", score[0])
            print("score inactive:", score[1])
        return np.array(loss), np.array(score)
