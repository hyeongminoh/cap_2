
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import Select_CIFAR10_Classes
# for image
import matplotlib.pyplot as plt
import numpy as np
import time

# Data
start_vect=time.time()

print('\n===> Loading Data...')

"""
- 데이터를 torch Tensor로 바꾸고 Normalization을 위해 transform.Compose 를 사용합니다.
- Compose 는 여러 transform 들을 chaining 합니다. 즉 여러 transform 진행합니다.
"""

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


car_trainset = Select_CIFAR10_Classes.DatasetMaker(
        [ Select_CIFAR10_Classes.get_class_i(x_train, y_train, classDict['car'])],
        transform_with_aug
    )

car_testset  = Select_CIFAR10_Classes.DatasetMaker(
        [ Select_CIFAR10_Classes.get_class_i(x_test , y_test , classDict['car'])],
        transform_no_aug
    )

kwargs = {'num_workers': 2, 'pin_memory': False}

trainloader   = DataLoader(car_trainset, batch_size=32, shuffle=True , **kwargs)
testloader    = DataLoader(car_testset , batch_size=32, shuffle=False, **kwargs)



# Model

print('\n===> Building Model - squeezenet1_0 - ...')
"""
- Cross Entropy loss 함수를 사용

"""

net = models.squeezenet1_0(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




# Training

print('\n===> Training Start')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)
if torch.cuda.device_count() > 1:
    print('\n===> Training on GPU!')
    net = nn.DataParallel(net)


epochs = 2 # dataset을 여러번 사용해 트레이닝을 시킵니다.

for epoch in range(epochs):
    print('\n===> epoch %d' % epoch)
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('\n===> Finished Training...')


#Save
savePath = "/data2/ohm/models"
torch.save(net.state_dict(), savePath)

# Test
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('\nAccuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



print("training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))