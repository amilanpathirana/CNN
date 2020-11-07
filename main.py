
print("Hey")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


batch_size=4
num_epochs=3 
learnning_rate=0.001

train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)

classes=["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

'''
dataiter=iter(train_loader)
images,lables=dataiter.next()
print(images[1])
print(lables[1])
'''


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        #nn.Conv2d()

    def forward(self,x):
        #print("X Shape :",x.shape)
        x3=self.pool(F.relu(self.conv1(x)))
        x4=self.pool(F.relu(self.conv2(x3)))
        x4=x4.view(-1,16*5*5)
        x5=F.relu(self.fc1(x4))
        x6=F.relu(self.fc2(x5))
        x7=self.fc3(x6)

        return x7

model=CNNNet().to(device)

criterion=nn.CrossEntropyLoss()
 
optimizer=torch.optim.SGD(model.parameters(),lr=learnning_rate)






steps=len(train_loader)

for epoch in range (num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #image size is 100x1x28x28   , need to convert this to 100x784
        
        images=images.to(device)
        labels=labels.to(device)
        #print(images.shape)
        outputs=model(images)

        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%2000==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {steps}, loss= {loss.item():.4f}')

print("finished Trainning")



#Test
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')














