import os
import shutil
import numpy as np
import pandas as pd
import glob
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from sklearn.metrics import make_scorer, precision_score, classification_report

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

@make_scorer
def avg_accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)*100

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

def load_dataset(dir, train_set=True):
    """
    Read in data and apply transformation.

    Args:
        dir (string): path to data folder
    Returns:
        data and labels (list).
    """
    data = []
    labels = []
    paths = []
    class_idx_to_label = {}
    classes = [name[31:] for name in glob.glob(dir + '/*')]
    classes = dict(zip(range(0,len(classes)), classes))

    if (train_set):
        for id, class_name in classes.items():
            class_idx_to_label[class_name] = id
            img_path_class = glob.glob(dir + class_name + '/*.jpg')
            labels.extend([id]*len(img_path_class))
            for filename in img_path_class:
                data.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    else:
        img_path_class = glob.glob(dir + '/*.jpg')
        for filename in img_path_class:
            data.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    
    test_path = dir
    test_names = os.listdir(test_path)
    for name in test_names:
        path = os.path.join(test_path, name)
        paths.append(path)
        
    return data, labels, paths, class_idx_to_label

def write_preds_to_file(file_name, image_paths, class_labels, class_to_label_dict):
    with open(file_name +'.txt', 'w') as file:
        for idx in range(len(image_paths)):
            path = image_paths[idx]
            line = str(path) + " " + list(class_to_label_dict.keys())[list(class_to_label_dict.values()).index(class_labels[idx])]
            file.write(line)
            file.write('\n')
    file.close()
    
    
def plot_cm(y_true, y_pred, labels, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    col_labels = labels
    cm = pd.DataFrame(cm, index=col_labels, columns=col_labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    figure = hm.get_figure()
    figure.savefig("out.png") 
