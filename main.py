from sup_cl import SupCL
from resnet_big import SupConResNet
import torch

numclass = 10
feature_extractor = SupConResNet()
img_size = 32
batch_size = 512
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 0.01
temperature = 0.1

model = SupCL(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, temperature)

# model.model.load_state_dict(torch.load('model/_.pkl'))

for i in range(10):
    model.beforeTrain()
    accuracy = model.train()
    model.afterTrain(accuracy)
