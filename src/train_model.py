import cv2
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset, ImageNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import PIL.Image as Image
import fs

from lib.binary_mat_utils import convert_image_data_to_one_zero_mat

GRAYSCALE_WHITE_THRESHOLD = 245

img = cv2.imread('./src/training_data/character_zou/zou_charactrer_prepared_0_test.png', cv2.IMREAD_UNCHANGED)

def read_data(folde_path): 
    data = []
    files = os.listdir(folde_path)    
    
    for file in files: 
        full_path = folde_path + '/' + file
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        data.append(convert_image_data_to_one_zero_mat(img, GRAYSCALE_WHITE_THRESHOLD))
    
    return data

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size = (5,5), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size = (5,5), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Linear(64 * 50, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

transform = transforms.Compose([transforms.Resize(48),
                                transforms.Grayscale(),
                                 transforms.ToTensor()])

trainset = ImageFolder('./src/training_data', transform=transform)
testset = ImageFolder('./src/training_data', transform=transform)

batch_size = 32

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size,
                                         shuffle=False)


classes = ["niu", "yang", "zou"]

images, labels = next(iter(trainloader))
# print(len(images))
# npimg = images[0].numpy()
# plt.imshow(images[0].permute(1, 2, 0))
# plt.show()

device = torch.device("cpu")
net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 200

for epoch in range(num_epochs):  # loop over the dataset multiple times
    train_loss = 0
    valid_loss = 0
    train_corrects = 0
    valid_corrects = 0

    for i, (images, labels) in enumerate(trainloader, 0):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_corrects += (labels == torch.max(outputs, 1).indices).sum().item()

    for i, (images, labels) in enumerate(testloader, 0): 
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        valid_corrects += (labels == torch.max(outputs, 1).indices).sum().item()

    print("Epoch [{}/{}]: Train loss = {:.4f}, Valid loss = {:.4f}, Train Acc = {:.4f}, Valid Acc = {:.4f}".format(epoch, num_epochs, train_loss/len(trainloader), valid_loss/len(testloader), train_corrects/(len(trainloader) * batch_size), valid_corrects/(len(testloader) * batch_size)))


torch.save(net, "./model/model.pt")

test_net = net.eval()

image1 = Image.open('./src/test_data/running_man/running_man_image_5_preprocessed_mirror_skeletonized_test.png')
image2 = Image.open('./src/test_data/running_man/running_man_image_5_preprocessed_mirror.png')
image1 = transform(image1).float().unsqueeze(0)
image2 = transform(image2).float().unsqueeze(0)
output1 = test_net(image1)
output2 = test_net(image2)

_, predicted1 = torch.max(output1.data, 1)
_, predicted2 = torch.max(output2.data, 1)

print(output1.data)
print(output2.data)

classes = ["niu", "yang", "zou"]
print("predicated 1: " + str(classes[predicted1]))
print("predicated 2: " + str(classes[predicted2]))


print('Finished Training and testing')