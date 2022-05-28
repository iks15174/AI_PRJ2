import torch
import torchvision.models as models

PretrainCNN = models.alexnet(pretrained=True)
PretrainCNN.classifier[6] = torch.nn.Linear(4096, 6)
