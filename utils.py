import os
import shutil
import numpy as np
import pandas as pd
from torchvision import datasets
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

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


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def write_preds_to_file(file_name, image_paths, class_labels, class_to_label_dict):
    with open(file_name +'.txt', 'w') as file:
        for idx in range(len(image_paths)):
            path = image_paths[idx].split("\\")[1]
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