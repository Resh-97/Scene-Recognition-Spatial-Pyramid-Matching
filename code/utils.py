import copy
import os
import shutil
import glob
import time
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, make_scorer, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from PIL import Image
from itertools import compress
from torchvision import datasets


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


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def write_preds_to_file(file_name, image_paths, class_labels, class_to_label_dict):
    with open(file_name +'.txt', 'w') as file:
        for idx in range(len(image_paths)):
            path = image_paths[idx].split("/")[3]
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


class CodeBook():
    def __init__(self, should_tune_code_book_size=False):
        self.feature_scaler = StandardScaler()
        self.sample_size = 500000       # for code book generation
        self.code_book_size = 600
        self.should_tune_code_book_size = should_tune_code_book_size
        
        
    def stack_descriptors(self, descriptors):
        descriptor_stack = descriptors[0][:]
        for descriptor in descriptors[1:]:
            descriptor_stack = np.vstack((descriptor_stack, descriptor))
        return descriptor_stack.astype(float)  
        
        
    def create_code_book(self, descriptors, should_tune_code_book_size, code_book_cluster_num_candidates):
        self.should_tune_code_book_size = should_tune_code_book_size
        stacked_descriptors = self.stack_descriptors(descriptors)
        normalised_descriptors = self.feature_scaler.fit_transform(stacked_descriptors)
        samples = normalised_descriptors[np.random.choice(normalised_descriptors.shape[0], self.sample_size, replace=False), :]
        
        # Choose cluster num for code book
        if (self.should_tune_code_book_size):
            samples_for_codebook = normalised_descriptors[np.random.choice(normalised_descriptors.shape[0], 10000, replace=False), :]
            self.code_book_size = self.tune_k(samples_for_codebook, code_book_cluster_num_candidates)
            
        self.code_book, _ = kmeans(samples, self.code_book_size, 1)
    
    
    def get_quantised_image_features(self, descriptors):
        im_features = np.zeros((len(descriptors), self.code_book_size), "float32")

        for i in range(len(descriptors)):
            transformed_descriptor = self.feature_scaler.transform(descriptors[i])
            codes, distances = vq(transformed_descriptor, self.code_book)
            for code in codes:
                im_features[i][code] += 1

        return im_features
    
    
    def tune_k(self, samples_for_codebook, candidate_values):
        weighted_ssd = {}

        for k in candidate_values:
            kmeans = KMeans(n_clusters=k, max_iter=100).fit(samples_for_codebook)
            weighted_ssd[k] = kmeans.inertia_/samples_for_codebook.shape[0]

        plt.figure()
        plt.plot(list(weighted_ssd.keys()), list(weighted_ssd.values()))
        plt.xlabel("Number of clusters")
        plt.ylabel("Avg weighted sum of squared distance")
        plt.savefig('clustering.jpg')

        return min(weighted_ssd, key=weighted_ssd.get)

"""
For training a PyTorch neural network.
"""
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

"""
Confusion matrix plot.
"""
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
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)