import os
import shutil
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
