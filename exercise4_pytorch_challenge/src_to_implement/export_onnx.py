import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model  # Import your model

epoch = int(sys.argv[1])

# TODO: Enter your model here
model_instance = model.ResNet()  # Create your ResNet model

crit = t.nn.BCELoss()
trainer = Trainer(model_instance, crit, cuda=t.cuda.is_available())
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
