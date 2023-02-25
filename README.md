# EVA8_Pytorch_Models

Main repository containing different models, main file used for training the models and utils file for miscelleneous functions like visualization, GradCAM, debugging etc. 

## Directory Structure
```
|── models
│       ├── resnet.py 
|       ├── custom_resnet.py 
|       └── model9_transformers.py
├── utils.py
├── main.py
└── README.md
```

### Models dir
Contains different models for e.g currently resnet.py contains Basic ResNet18/34 models

### main.py
Functions needed o train and test a model.

### utils.py
Image Transformations/Augmentation, GradCAM, Visualize Dataset, Visualize Augmentations, Display Misclassified images etc. 
