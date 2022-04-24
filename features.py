import numpy as np
from PIL import Image
import cv2
from torch import flatten, from_numpy
from torchvision import transforms
from sklearn.cluster import KMeans


##-------------------------------------- FEATURE EXTRACTION CLASSES -----------------------## 
class ZeroMeanTransform:
    def __call__(self, img):
        """
        Convert image to numpy array, 
        do any preprocessing you like then 
        convert back to the PIL Image.

        Args:
            img(PIL.Image): input image 
        Returns:
            (PIL.Image) processed image
        """
        x = np.array(img, dtype=np.float32)
        mean = np.mean(img)
        return Image.fromarray((x - mean))

class UnitLenTransform:
    def __call__(self, img):
        """
        Convert image to numpy array, 
        do any preprocessing you like then 
        convert back to the PIL Image.

        Args:
            img (PIL.Image): input image 
        Returns:
            (PIL.Image) processed image
        """
        x = np.array(img)
        cv2.normalize(x, x, alpha=1, dtype=cv2.CV_32F)
        return Image.fromarray(x)


class GreyScaleToRGB:
    def __call__(self, img):
        rgb_img = img.convert('RGB')
        return rgb_img


class DenseSIFT:
    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, img_tensor):
        img = img_tensor.numpy()
        # convert image to an 8 bit it for input to SIFT
        img8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #img = Image.fromarray(img_array[0, 0, :])

        sift = cv2.SIFT_create()
        step_size = 5
        # extract key points
        kp = [cv2.KeyPoint(x, y, self.step_size) for y in range(0, img8bit.shape[2], step_size) 
                                        for x in range(0, img8bit.shape[1], step_size)]
        # extract dense features, NOTE: first dimension is batch size so drop
        kp, descr = sift.compute(img8bit[0,:,:], kp)
        return from_numpy(descr).unsqueeze(0) # add back batch size dimension




##------------------------------------------------------------------------------------------##

##-------------------------------------- TRANSFORM FUNCTIONS -------------------------------##

def cluster_features(img_descs, training_idxs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    # img_descs = gen_sift_features(labeled_img_paths)
    #
    # # Generate indexes of training rows
    # total_rows = len(img_descs)
    # training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, percent_test, percent_val)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # Cluster descriptors to get codebook
    print('Using clustering model %s...' % repr(cluster_model))
    print('Clustering on training set to get codebook of %i words' % n_clusters)

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('done generating BoW histograms.')

    return X, cluster_model


def get_bovw(data, num_clusters):
    cluster_model = KMeans(n_clusters=num_clusters)
    print(data.shape)
    # first we need to stick all our images together into one big array
    stacked_descr = np.concatenate([img_array for img_array in data[:,0,:,:]])
    print(stacked_descr.shape)
    cluster_model.fit(stacked_descr)
    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(img_descr[0,:,:]) for img_descr in data]
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=num_clusters) for clustered_words in img_clustered_words])
    X = img_bow_hist
    return X



def run1_transforms(resize=16, crop=240):
    """
    Compute tiny image feature.

    Args:
        resize (int): the dimensions of the tiny image.
        crop (int): the dimensions of the center cropped array.
    Returns:
        (numpy.array): Tiny image flattened numpy array. 
            Dimensions are (num input samples, resize x resize) 
    """
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(255),
                               transforms.CenterCrop(crop),
                               transforms.Resize(resize),
                               ZeroMeanTransform(),
                               UnitLenTransform(),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: flatten(x))])


                
def run3_SIFT_transforms():
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((255,255)),
                               transforms.ToTensor(),
                               DenseSIFT(step_size=5)])

def run3_transforms(resize=16, crop=240):
    """
    Compute tiny image feature.

    Args:
        resize (int): the dimensions of the tiny image.
        crop (int): the dimensions of the center cropped array.
    Returns:
        (numpy.array): Tiny image flattened numpy array. 
            Dimensions are (num input samples, resize x resize) 
    """
    return transforms.Compose([
        GreyScaleToRGB(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
##-------------------------------------------------------------------------------------------##