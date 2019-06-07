import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd

class CustomDataset(torch.utils.data.Dataset):
    """
        참고)https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas
    """
    def __init__(self, csv_path):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # first column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        ## Get m_label from the pandas df
        single_m_label = self.label_arr[index]
        ## Read each 20*20 pixels and reshape the 1D array ([400]) to 2D array ([20,20]) 
        img_as_np = np.asarray(self.data_info.iloc[index][1:]).reshape(20,20).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'RGB' is for 3 channel
        img_as_img = Image.fromarray(img_as_np)
        #img_as_img.show()
        img_as_img = img_as_img.convert('RGB')
        ## Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        ## Return image and the label
        return (img_as_tensor, single_m_label)

    def __len__(self):
        return self.data_len
    
if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])
    # Define custom dataset
    custom_dataset = \
        CustomDataset('Total_mini.csv')
    
    # Splic train/test set
    train_size = int(0.8 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])
    print("**train len: ",len(train_dataset),"/test len: ", len(test_dataset), "/total len: ", len(custom_dataset))
        
    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=3,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=3,
                                                    shuffle=False)
    
    classes_mini=("ARIAL","AGENCY","ARIAL BLACK","ARIAL NARROW","ARIAL ROUNDED MT BOLD","ARIAL_scanned")

    net = models.squeezenet1_0(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # GPU
    print('\n===> Training Start')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print('\n===> Training on GPU!')

    # Training
    epochs = 1 # dataset을 여러번 사용해 트레이닝을 시킵니다.

    for epoch in range(epochs):
        print('\n===> epoch %d' % epoch)
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward +  optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #loss.backward()    #백워드는 생략
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    #Test
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            images = images.to(device)
            labels = labels.to(device)
            print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted[0])
            total += labels.size(0)            
            correct += (predicted == labels).sum().item()
    print("total: ", total, "correct: ",correct)

    print('Accuracy of the network on the 5449 test images: %d %%' % (100 * correct / total))