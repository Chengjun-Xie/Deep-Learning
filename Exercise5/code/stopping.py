import math


class EarlyStoppingCallback:

    def __init__(self, patience):
        # initialize all members you need
        self.patience = patience
        self.pre_loss = 0.0
        self.counter = 0

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss > self.pre_loss:
            self.counter += 1
        else:
            self.counter = 0
            self.pre_loss = current_loss

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.counter > self.patience

