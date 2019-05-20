#### keep it and wait next update

import torch
import torch.nn as nn
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([transforms.Resize((240,240)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

dataset = datasets.VOCDetection("/data2/ohm/datasets", year='2012', image_set='trainval', 
                                download=False, transform=transform)

dataloader = torch.utils.data.DataLoader( dataset = dataset, batch_size=1, shuffle=True, num_workers=1)

#X, y = next(iter(dataloader))
#print(X.shape)

for X, y in iter(dataloader):
	#print(X.shape)
	if isinstance(y['annotation']['object'],list) == True:
		#if y['annotation']['object'][0] == 'car' :
		for i in y['annotation']['object']:
			print('l' , i)
	else:
		#if y['annotation']['object']['name'] == 'car' :
		for i in y['annotation']['object'].keys():
			print('k', i, type(y['annotation']['object'][i]), y['annotation']['object'][i])

#for X,y in iter(dataloader):
#    print(X.shape)