import numpy as np


class base:
    """
    All layers need to inherit from this ”base-layer” so refactor them accordingly.
    """
    def __init__(self, phase='train'):
        self.phase = None


class Phase:
    train = 'train'
    test = 'test'


