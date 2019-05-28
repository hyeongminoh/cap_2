import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np
import time

# Data
start_vect=time.time()
print('===> Loading Data...')

transform = transforms.Compose([transforms.Resize((240,240)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


# Model

print('===> Building Model - squeezenet1_0 - ...')

net = models.squeezenet1_0(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



# Training

"""
GPU 사용하기
"""

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
savePath = "/data2/ohm/cap_2/models"
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

print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))




print("training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))