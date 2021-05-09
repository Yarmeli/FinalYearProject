import os, torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FoodDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class FoodCNN(nn.Module):
    
    def __init__(self):
        super(FoodCNN, self).__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*55*55, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=17)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ImageClassModel = FoodCNN().to(device)

def train_model():
    
    train_csv = "Dataset/train_labels.csv"
    test_csv = "Dataset/test_labels.csv"

    train_imgdir = "Dataset/Train"
    test_imgdir = "Dataset/Test"
    
    batch_size = 8
    size = 224

    transform=transforms.Compose([transforms.Resize(size),
                                  transforms.CenterCrop(size),
                                  transforms.ToTensor()])

    train_set = FoodDataset(train_csv, train_imgdir, transform=transform)
    test_set = FoodDataset(test_csv, test_imgdir, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    num_epochs = 5
    train_losses = []
    valid_losses = []
    
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(ImageClassModel.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        
        ImageClassModel.train()
        for data, target in train_loader:
            
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = ImageClassModel(data.float())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        ImageClassModel.eval()
        for data, target in test_loader:
            
            data = data.to(device)
            target = target.to(device)
            
            output = ImageClassModel(data.float())
            loss = criterion(output, target)
            
            valid_loss += loss.item() * data.size(0)
            
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(test_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
                
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))