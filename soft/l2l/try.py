import learn2learn as l2l
import torchvision as tv
from fc100 import FC100
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

train_dataset = FC100(
    root='~/data',
    transform = tv.transforms.ToTensor(),
    mode='train',
    download=True,
 )
 
train_dataset = l2l.data.MetaDataset(train_dataset)
# train_transforms = [ 
#     l2l.data.transforms.FusedNWaysKShots(train_dataset),
#     l2l.data.transforms.LoadData(train_dataset),
#     l2l.data.transforms.RemapLabels(train_dataset),
#     l2l.data.transforms.ConsecutiveLabels(train_dataset),
# ]
# train_tasks = l2l.data.TaskDataset(
#     train_dataset,
#     task_transforms=train_transforms,
#     num_tasks = 20000
# )
# #features = l2l.vision.models.ConvBase