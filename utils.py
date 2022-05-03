import os
import shutil
import numpy as np
import glob
import cv2
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
            path = image_paths[idx].split("\\")[3]
            line = str(path) + " " + list(class_to_label_dict.keys())[list(class_to_label_dict.values()).index(class_labels[idx])]
            file.write(line)
            file.write('\n')
    file.close()
    
    
def write_classification_report_to_file(file_name, true_classes, predicted_classes, target_names):
    report = classification_report(true_classes, predicted_classes, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report = df_report.round(2)
    report_latex = df_report.to_latex(caption='Run#2 classification report.', label='tab::run2report')
    with open(file_name +'.txt', 'w') as file:
        file.write(report_latex)
    file.close()