import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import resnet

epoch = int(sys.argv[1])
#TODO: Enter your model here
model = resnet.ResNet18()
crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
