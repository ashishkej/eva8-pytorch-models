import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torch.utils.data import Dataset



def gradcam_vis(model,target_layers, misclf, classes,plot_size=(4,5)):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        img_model, img, label, pred = misclf[i-1]
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=img_model.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img) # showing the plot
        plt.imshow(grayscale_cam, cmap='magma', alpha=0.65)

    plt.tight_layout()
    plt.show()


def convert_image_np(inp, mean, std):
    """Convert normalized tensor to numpy image for display.

    Args:
        inp (tensor): Tensor image
        mean(np array): numpy array of mean of dataset
        std(np array): numpy array of standard deviation of dataset

    Returns:
        np array: a numpy image
    """

    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def plot_data(data, rows, cols):
    """Randomly plot the images from the dataset for vizualization

    Args:
        data (instance): torch instance for data loader
        rows (int): number of rows in the plot
        cols (int): number of cols in the plot
    """
    figure = plt.figure(figsize=(cols*2,rows*3))
    for i in range(1, cols*rows + 1):
        k = np.random.randint(0,50000)
        figure.add_subplot(rows, cols, i) # adding sub plot

        img, label = data[k]
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Class: {label} '+data.classes[label])

    plt.tight_layout()
    plt.show()


def show_aug_images(exp, aug_dict, ncol=6):
  nrow = len(aug_dict)

  fig, axes = plt.subplots(ncol, nrow, figsize=( 3*nrow, 15), squeeze=False)
  for i, (key, aug) in enumerate(aug_dict.items()):
    for j in range(ncol):
      ax = axes[j,i]
      if j == 0:
        ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
      else:
        image, label = exp[j-1]
        if aug is not None:
          transform = A.Compose([aug])
          image = np.array(image)
          image = transform(image=image)['image']
          
        ax.imshow(image)
        ax.set_title(f'{exp.classes[label]}')
        ax.axis('off')

  plt.tight_layout()
  plt.show()
    

def plot_misclassified(model, test_loader, classes, device, dataset_mean, dataset_std, no_misclf=20, plot_size=(4,5), return_misclf=False):
    """Plot the images are wrongly clossified by model

    Args:
        model (instance): torch instance of defined model (pre trained)
        test_loader (instace): torch data loader of testing set
        classes (dict or list): classes in the dataset
                if dict:
                    key - class id
                    value - as class name
                elif list:
                    index of list correspond to class id and name
        device (str): 'cpu' or 'cuda' device to be used
        dataset_mean (tensor or np array): mean of dataset
        dataset_std (tensor or np array): std of dataset
        no_misclf (int, optional): number of misclassified images to plot. Defaults to 20.
        plot_size (tuple): tuple containing size of plot as rows, columns. Defaults to (4,5)
        return_misclf (bool, optional): True to return the misclassified images. Defaults to False.

    Returns:
        list: list containing misclassified images as np array if return_misclf True
    """
    count = 0
    k = 0
    misclf = list()
  
    while count<no_misclf:
        img_model, label = test_loader.dataset[k]
        pred = model(img_model.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img_model, dataset_mean, dataset_std)
            misclf.append((img_model, img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        _, img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if return_misclf:
        return misclf

def get_mean_and_std(exp_data):
    '''Calculate the mean and std for normalization'''
    print(' - Dataset Numpy Shape:', exp_data.shape)
    print(' - Min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - Max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - Mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - Std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - Var:', np.var(exp_data, axis=(0,1,2)) / 255.)
    return np.mean(exp_data, axis=(0,1,2)) / 255., np.std(exp_data, axis=(0,1,2)) / 255.

class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train= True):
        self.image_list = image_list
        self.aug = A.Compose({
            A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by default
                A.RandomCrop(32,32)]),
            A.HorizontalFlip(),
            A.CoarseDropout(1, 8, 8, 1, 8, 8,fill_value=0.473363, mask_fill_value=None),
            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        })

        self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        })
        self.train = train
            
    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        
        image, label = self.image_list[i]
        
        if self.train:
            #apply augmentation only for training
            image = self.aug(image=np.array(image))['image']
        else:
            image = self.norm(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image, dtype=torch.float), label