import os, torch, time, copy
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from helpers import Debug

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
        self.features = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=44, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            
            nn.Conv2d(in_channels=44, out_channels=88, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=88 * 55 * 55, out_features=100), 
            nn.ReLU(),
            nn.Dropout(0.25), 
            
            nn.Linear(in_features=100, out_features=17)
        )

    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ImageClassModel = FoodCNN().to(device)
num_workers = 2


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    print("Training the model....")
    since = time.time()

    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []

    # Keep track of the best weights 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        last_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['training', 'validation']:
            # Set the model either to training or evaluate mode
            if phase == 'training': 
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                # reset the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'training'):
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Update weights
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # get loss and accuracy for each batch
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Compare Accuracy with the best model
            if phase == 'validation' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict()) # Save current weights
            if phase == 'validation':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            else:
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            
        time_elapsed = time.time() - last_epoch_time
        print('Epoch duration {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))


    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss, train_loss, val_acc, train_acc


def SetupTrainTestLoaders():    
    Debug("Model", "Initializing Datasets and Dataloaders...")    
        
    # Set Batch size and resize size
    batch_size = 32
    size = 224
    
    Debug("Model", f"Using Batch size of '{batch_size}'")
    
    # Setup label files and image directories
    train_csv = "Dataset/train_labels.csv"
    test_csv = "Dataset/test_labels.csv"

    train_imgdir = "Dataset/Train"
    test_imgdir = "Dataset/Test"
    
    train_tranform=transforms.Compose([transforms.RandomResizedCrop(size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
    
    val_transform=transforms.Compose([transforms.Resize(size),
                                      transforms.CenterCrop(size),
                                      transforms.ToTensor()])
    
    # Create training and validation datasets
    train_set = FoodDataset(train_csv, train_imgdir, transform=train_tranform)
    test_set = FoodDataset(test_csv, test_imgdir, transform=val_transform)
    
    # Create the dataloader that will return 'batch_size' items at once (e.g. 8 items per iteration)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    dataloader_dict = { 'training' : train_loader, 'validation': test_loader }
    return dataloader_dict


def TrainFoodCNN():
    global ImageClassModel
    
    dataloader_dict = SetupTrainTestLoaders()
        
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    Debug("Model", f"Using Learning Rate of '{learning_rate}'")
    optimizer = torch.optim.Adam(ImageClassModel.parameters(), lr=learning_rate)
    
    Debug("Model", f"Using num_workers: '{num_workers}'")
    
    ImageClassModel, valid_loss, train_loss, val_acc, train_acc = train_model(ImageClassModel, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
    
    torch.cuda.empty_cache()  
    
    plt.plot(valid_loss, label="validation loss")
    plt.plot(train_loss, label="training loss")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(val_acc, label="validation accuracy")
    plt.plot(train_acc, label="training accuracy")
    plt.legend()
    plt.show()
    


def output_label(label):
    output_mapping = {
                0: 'Apple',
                1: 'Banana',
                2: 'Bean',
                3: 'Bread',
                4: 'Carrot',
                5: 'Cheese',
                6: 'Cucumber',
                7: 'Egg',
                8: 'Grape',
                9: 'Kiwi',
                10: 'Onion',
                11: 'Orange',
                12: 'Pasta',
                13: 'Pepper',
                14: 'Sauce',
                15: 'Tomato',
                16: 'Watermelon'
                }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]
   

def plotImage(image, label=None):
    if label is not None:
        if type(label) != str: # Label is a number, get the correct label name
            label = output_label(label)
        plt.title(label)
            
    img = np.transpose(image.squeeze(), (1,2,0)) # Change from (CxHxW) to (HxWxC) , C = colors, H = heigth, W = width
    plt.imshow(img)
    plt.show()


def EvaluateOnData(model, csvFile, imgdir):
    
    # Set Batch size and resize size
    batch_size = 32
    size = 224

    # Transforms
    transform=transforms.Compose([transforms.Resize(size),
                                  transforms.CenterCrop(size),
                                  transforms.ToTensor()])
    
    # Create the set
    dataset = FoodDataset(csvFile, imgdir, transform=transform)

    img_dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    images, labels = iter(img_dataLoader).next()
    
    Debug("Evaluate","Sample Image:")
    plotImage(images[0], labels[0])
    
    cm_predicted, cm_target = [], []
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in img_dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            cm_predicted.extend(predicted.tolist())
            cm_target.extend(labels.tolist())

    print('Accuracy of the model is: {:.4f} %'.format(100 * correct / total))
    
    cm = confusion_matrix(cm_target, cm_predicted)
    plt.imshow(cm)


def LoadSavedModel(file):
    print(ImageClassModel.load_state_dict(torch.load(file)))
    ImageClassModel.eval()
    print(f"Loaded model weights from '{file}'")
       

def SaveCurrentModel(file):
    print(f"Saving current model to '{file}'")
    torch.save(ImageClassModel.state_dict(), file)
   


