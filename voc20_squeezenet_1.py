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

print('===> Loading Data...')

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

testset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)


# Model

print('===> Building Model - squeezenet1_0 - ...')

net = models.squeezenet1_0(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def make_label(labels):
        if isinstance(labels['annotation']['object'],list) == True:
            for label_object in labels['annotation']['object']:
                if(label_object['name'] == ['car']):
                    return torch.tensor([1])
        else:
            dict_object = labels['annotation']['object']
            if(dict_object['name'] == ['car']):
                    return torch.tensor([1])
        return torch.tensor([0])

# Training

"""
GPU 사용하기
"""

print('\n===> Training Start')
start_vect=time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net.to(device)

if torch.cuda.device_count() > 1:
    print('\n===> Training on GPU!')
    net = nn.DataParallel(net)


epochs = 10 # dataset을 여러번 사용해 트레이닝을 시킵니다.

for epoch in range(epochs):
    print('\n===> epoch %d' % epoch)
    running_loss = 0.0

    for i,data in enumerate(trainloader,0):
        # get the inputs
        inputs, labels = data
        real_label = make_label(labels)
        inputs, real_label = inputs.to(device), real_label.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, real_label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('\n===> Finished Training...')


print("\n===> Training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

#Save
savePath = "/models"
torch.save(net.state_dict(), savePath)

# Test
print('\n===> Testing...')
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        real_label = make_label(labels)
        images = images.to(device)
        real_label = real_label.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += real_label.size(0)
        correct += (predicted == real_label).sum().item()

print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))



