import os, glob
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image

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
            image = self.transform(image)
            label = self.transform(label)
            
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
