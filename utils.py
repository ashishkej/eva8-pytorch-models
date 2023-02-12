import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import torch 
from torch.utils.data import Dataset

import cv2



def gradcam_vis(model,target_layers, input_tensor, target_class, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    fig, (ax1, ax2) = plt.subplots(figsize=(20,20), ncols=2)

    ax1.imshow(rgb_img)
    ax2.imshow(rgb_img)
    ax2.imshow(grayscale_cam, cmap='magma', alpha=0.7)


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


def plot_data(data, rows, cols, lower_value, upper_value):
    """Randomly plot the images from the dataset for vizualization

    Args:
        data (instance): torch instance for data loader
        rows (int): number of rows in the plot
        cols (int): number of cols in the plot
        lower_value (int): lower value of the dataset for plotting in a particular interval. 0 for starting index
        upper_value (int): upper value for plotting in a particular interval. len of dataset for last index index
    """
    figure = plt.figure(figsize=(cols*2,rows*3))
    for i in range(1, cols*rows + 1):
        k = np.random.randint(lower_value,upper_value)
        figure.add_subplot(rows, cols, i) # adding sub plot

        img, label = data.dataset[k]
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Class: {label}')

    plt.tight_layout()
    plt.show()


def plot_aug(aug_dict, data, ncol=6):
    """Vizualize the image for the augmentations to be applied over dataset

    Args:
        aug_dict (dict): dictionary key as name of augmentation to applied (str) and 
                                    value as albumentations aug function
        data (instance): torch instance for data loader
        ncol (int, optional): number of cols in the plot. Defaults to 6.
    """
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
                image, label = data.dataset[j-1]
                if aug is not None:
                    transform = A.Compose([aug])
                    image = np.array(image)
                    image = transform(image=image)['image']
                
                ax.imshow(image)
                #ax.set_title(f'{data.classes[label]}')
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
        img, label = test_loader.dataset[k]
        pred = model(img.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img, dataset_mean, dataset_std)
            misclf.append((img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if return_misclf:
        return misclf


def plot_learning_curves(history=None, from_txt=False, plot_lr_trend=False):
    """Plot Test & Train Learning Curves of model

    Args:
        history (tuple, optional): tuple of list contraing (training_acc, training_loss, testing_acc, testing_loss)
                            Note- in specific order only. Defaults to None. if information is dumped to txt file
        from_txt (bool or list, optional): List of path to (training_acc, training_loss, testing_acc, testing_loss) txt files
                            Note- give path in specific order only. Defaults to False.
        plot_lr_trend (bool or list or str, optional): List if plot learning curve trend form list 
                                                       str- plot from txt file, path to txt file. 
                                                       Defaults to False. (dont plot)
    """
    if from_txt:
        history = []
        for path in from_txt:
            history.append([float(i) for i in open(path).read().strip().split("\n")])
    

    fig, axs = plt.subplots(1,2,figsize=(16,7))
    axs[0].set_title('LOSS')
    axs[0].plot(history[1], label='Train')
    axs[0].plot(history[3], label='Test')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].plot(history[0], label='Train')
    axs[1].plot(history[2], label='Test')
    axs[1].legend()
    axs[1].grid()

    plt.show()

    if plot_lr_trend:
        if from_txt:
            lr_trend = [float(i) for i in open(plot_lr_trend).read().strip().split("\n")]
        else:
            lr_trend = plot_lr_trend
        
        plt.plot(lr_trend)
        plt.title('Learning Rate Change During Training')
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.grid()
        plt.show()


def get_mean_and_std(exp_data):
    '''Calculate the mean and std for normalization'''
    print(' - Dataset Numpy Shape:', exp_data.shape)
    print(' - Min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - Max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - Mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - Std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - Var:', np.var(exp_data, axis=(0,1,2)) / 255.)
    return np.mean(exp_data, axis=(0,1,2)) / 255.), np.std(exp_data, axis=(0,1,2)) / 255.)

class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train= True):
        self.image_list = image_list
        self.aug = A.Compose({
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by default
                A.RandomCrop(32,32)]),
            A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363, mask_fill_value=None),
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