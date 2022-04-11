import os
import shutil
import numpy as np
from torchvision import datasets
from sklearn.metrics import make_scorer, precision_score

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

    dataset = datasets.ImageFolder(dir, transform)
    return dataset