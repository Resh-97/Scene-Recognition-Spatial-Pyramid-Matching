import os
import shutil
import numpy as np
from torchvision import datasets
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler


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


def stack_descriptors(descriptors):
    descriptors = descriptors[1]
    
    for descriptor in descriptors[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    print("Descriptors stacked successfully!")
    return descriptors.astype(float)


def normalise_data(data):
    scaler = StandardScaler()
    normalised_descriptors = scaler.fit_transform(descriptors_float)
    return normalised_descriptors, scaler

def create_code_book(descriptors, code_book_size):
    sample_size = 500000    # for code book generation
    feature_scaler = StandardScaler()
    
    stacked_descriptors = stack_descriptors(descriptors)
    normalised_descriptors = feature_scaler.fit_transform(stacked_descriptors)
    samples = normalised_descriptors[np.random.choice(normalised_descriptors.shape[0], sample_size, replace=False), :]
    code_book, variance = kmeans(samples, code_book_size, 1)
    return code_book, feature_scaler


def get_quantised_image_features(descriptors, code_book_size, code_book, feature_scaler):
    im_features = np.zeros((len(descriptors), code_book_size), "float32")
    
    for i in range(len(descriptors)):
        transformed_descriptor = feature_scaler.transform(descriptors[i])
        codes, distances = vq(transformed_descriptor, code_book)
        for code in codes:
            im_features[i][code] += 1
    
    return im_features