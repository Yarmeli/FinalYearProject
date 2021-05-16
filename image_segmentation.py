import os, glob, torch, time, copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from helpers import Debug

class SegmentationDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        super(SegmentationDataset, self).__init__()
        self.img_files = sorted(glob.glob(os.path.join(folder_path,'Images','*.*')))
        self.label_files = sorted(glob.glob(os.path.join(folder_path,'Labels','*.*')))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        
        image = Image.open(img_path)
        label = Image.open(label_path)
        
        
        if self.transform:
            # Combine image and label pictures to apply the transformation on both (e.g. RandomHorizontalFlip)
            image_np = np.asarray(image)
            label_np = np.asarray(label)
            
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np

            # Convert to PIL Image
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transform(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            label = label.long().squeeze()
            
        return image, label


def DeepLabModel(keep_feature_extract=False, use_pretrained=True):
    # Download and load the model
    DeepLabV3Model = models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained, progress=True)
    DeepLabV3Model.aux_classifier = None # Remove the Aux layer
    
    # keep_feature_extract = True  - Use already learnt features and only perform classification
    # keep_feature_extract = False - Retrain the model to detect custom features
    if keep_feature_extract: 
        for param in DeepLabV3Model.parameters():
            param.requires_grad = False

    # Change the classifier to have 19 classes - 17 for the food items, 1 for the thumb and 1 for background
    DeepLabV3Model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 19)

    return DeepLabV3Model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ImageSegModel = DeepLabModel().to(device)
num_workers = 2

# Obtained these values from the documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



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
            # Set the model either to training or evaluation mode
            if phase == 'training': 
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_iou_means = []

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                
                # Only process batch_size > 1
                if images.shape[0] == 1:
                    continue
                
                # reset the gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'training'):
                    
                    outputs = model(images)['out'] # only interested in the 'out' values - ignore 'aux'
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Update weights
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # calculate accuracy - Intersection over Union (IoU)
                ious = []
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                for classes in range(1, 19): # Ignore background class 0
                    pred_idxs = preds == classes
                    target_idxs = labels == classes
                    intersection = (pred_idxs[target_idxs]).long().sum().data.cpu().item()
                    union = pred_idxs.long().sum().data.cpu().item() + target_idxs.long().sum().data.cpu().item() - intersection
                    if union > 0:
                        ious.append(float(intersection) / float(max(union, 1)))
                  
                iou_mean = np.array(ious).mean()
                
                # track accuracy and loss values
                running_iou_means.append(iou_mean)
                running_loss += loss.item() * images.size(0)
                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if running_iou_means:
                epoch_acc = np.array(running_iou_means).mean()
            else:
                epoch_acc = 0.

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
        
        # Save progress every 5 epochs
        if epoch + 1 % 5 == 0:
            SaveImageSegModel(f"Dataset/ImageSegModel_checkpoint_{epoch:04}.pt")
        print()


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
    train_imgdir = "Dataset/ImageSegmentation/Train"
    test_imgdir = "Dataset/ImageSegmentation/Test"
    
    train_tranform=transforms.Compose([transforms.RandomResizedCrop(size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
    
    val_transform=transforms.Compose([transforms.Resize(size),
                                      transforms.CenterCrop(size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])
    
    
    # Create training and validation datasets
    train_set = SegmentationDataset(train_imgdir, transform=train_tranform)
    test_set = SegmentationDataset(test_imgdir, transform=val_transform)
    
    # Create the dataloader that will return 'batch_size' items at once (e.g. 8 items per iteration)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    dataloader_dict = { 'training' : train_loader, 'validation': test_loader }
    return dataloader_dict


def TrainImageSegmentation():
    global ImageSegModel
    
    dataloader_dict = SetupTrainTestLoaders()
        
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    Debug("Model", f"Using Learning Rate of '{learning_rate}'")
    optimizer = torch.optim.Adam(ImageSegModel.parameters(), lr=learning_rate)
    
    Debug("Model", f"Using num_workers: '{num_workers}'")
    
    ImageSegModel, valid_loss, train_loss, val_acc, train_acc = train_model(ImageSegModel, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
    
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
    

def LoadSavedImageSegModel(file = "Dataset/ImageSegModel.pt"):
    Debug("Load Seg Model", ImageSegModel.load_state_dict(torch.load(file)))
    ImageSegModel.eval()
    Debug("Load Seg Model", f"Loaded model weights from '{file}'")


def SaveImageSegModel(file = "Dataset/ImageSegModel.pt"):
    torch.save(ImageSegModel.state_dict(), file)
    Debug("Save Seg Model", f"Saved current model to '{file}'")