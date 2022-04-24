import copy
import os
import shutil
import time
import torch
import numpy as np
from PIL import Image
from itertools import compress
from torchvision import datasets
from sklearn.metrics import make_scorer, precision_score


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0] #NOTE removed [0] to index only path
        path_class_idx = self.imgs[index][1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,) + (path_class_idx,))
        return tuple_with_path


# Decorator turns into scorer object to be used with 
# grid search.
@make_scorer
def avg_precision(y_true, y_pred):
    """
    Average precision metric.
    Computes precision for each class independently
    then take average.

    Args:
        y_true (numpy.array): array of true y values. 
        y_pred(numpy.array): array of predicted y values.

    Returns:
        (float) average precision.
    
    """
    precision=[]
    for idx, label in enumerate(np.unique(y_true)):
        precision.append(precision_score(y_true == label, y_pred == label))
    return np.mean(precision)


def create_dataset(dir, transform=None, labeled=True):
    """
    Read in data and apply transformation.

    Args:
        dir (string): path to data folder
        transform (torchvision.Transform): transform to apply to data.
        labeled (bool): if data is not contained within subfolders indicating
            labels then create dummy folder and move data.
    Returns:
        (torch.Dataset) transformed data.
    """
    if not labeled:
        # get file names in folder
        file_names = os.listdir(dir)
        # create sub folder and move images
        target_dir = dir + '/dummy_class/'
        os.makedirs(target_dir)
        for file_name in file_names:
            shutil.move(os.path.join(dir, file_name), target_dir)

    #dataset = datasets.ImageFolder(dir, transform)
    dataset = ImageFolderWithPaths(dir, transform)
    return dataset


def show_misclassified(preds, true, paths, class_label_to_idx, k=5):
    mask = np.not_equal(preds,true)
    # find first k incorrect prediction in paths and load image and display
    idx=0
    displayed=0
    class_labels = list(class_label_to_idx.keys())
    class_ids = list(class_label_to_idx.values())
    mis_preds = list(compress(preds, mask))
    mis_labels = list(compress(true, mask))
    mis_paths = list(compress(paths, mask))
    while (idx < len(paths)) & (displayed <= k):
        print(mis_paths[idx])
        image = Image.open(mis_paths[idx])
        print(f"Predicted: {class_labels[class_ids.index(mis_preds[idx])]}\n")
        print(f"True: {class_labels[class_ids.index(mis_labels[idx])]}\n")
        image.show()
        displayed+=1
        print(f"{mis_paths[idx]}\n")
        idx+=1


def get_percent_misclassified(preds, true, class_label_to_idx):
    class_labels = list(class_label_to_idx.keys())
    class_ids = list(class_label_to_idx.values())
    mask = np.not_equal(preds,true)
    # misclassified labels
    mis_labels = true[mask]
    percent_misclassified = {}
    values, counts = np.unique(true, return_counts=True)
    for label, count in zip(values, counts):
        percent_misclassified[class_labels[class_ids.index(label)]] =  np.sum(np.where(mis_labels == label, 1,0))/ count
    return percent_misclassified


def test_paths(labels, paths, path_classes,  class_label_to_idx):
    class_labels = list(class_label_to_idx.keys())
    class_ids = list(class_label_to_idx.values())
    i=0
    for label, path, path_class in zip(labels, paths, path_classes):
        print(f"Label y: {class_labels[class_ids.index(label)]}, \
            path: {path} \
            label folder: {class_labels[class_ids.index(path_class)]}\n")
        if i>=10:
            break

def train_model(model, 
                data,
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            data_tuple = data[phase]
            for inputs, labels in zip(data_tuple[0], data_tuple[1]):
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data[phase][0].shape[0]
            epoch_acc = running_corrects.double() / data[phase][0].shape[0]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model