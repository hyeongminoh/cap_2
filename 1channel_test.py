import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np
import time

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class MySqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(MySqueezeNet, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 46, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(46, 16, 64, 64),
            #Fire(128, 16, 64, 64),
            Fire(64, 32, 128, 128),
            #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #Fire(256, 32, 128, 128),
            Fire(128, 48, 192, 192),
            #Fire(384, 48, 192, 192),
            Fire(256, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

# Data

print('===> Loading Data...')

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((240,240)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

testset = torchvision.datasets.VOCDetection("/data/datasets", year='2012', image_set='trainval', 
                                download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)


# Model

print('===> Building Model - My squeezenet - ...')

net = MySqueezeNet()
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
    


epochs = 3 # dataset을 여러번 사용해 트레이닝을 시킵니다.

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
savePath = "data/models"
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

print(correct)
print(total)
print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))



