from torchvision import models


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
