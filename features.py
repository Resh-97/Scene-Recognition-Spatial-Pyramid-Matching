import numpy as np
from PIL import Image
import cv2
from torch import flatten, from_numpy
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans


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
        # extract key points
        kp = [cv2.KeyPoint(x, y, self.step_size) for y in range(0, img8bit.shape[2], self.step_size) 
                                        for x in range(0, img8bit.shape[1], self.step_size)]
        # extract dense features, NOTE: first dimension is batch size so drop
        kp, descr = sift.compute(img8bit[0,:,:], kp)
        return from_numpy(descr).unsqueeze(0) # add back batch size dimension


class BOW_w_DenseSIFT:
    def __init__(self, step_size, num_clusters):
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

def get_bovw(data, num_clusters):
    cluster_model = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000)
    print(data.shape)
    # first we need to stick all our images together into one big array
    stacked_descr = np.concatenate([img_array for img_array in data[:,0,:,:]])
    print(stacked_descr.shape)
    cluster_model.fit(stacked_descr)
    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(img_descr[0,:,:]) for img_descr in data]
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=num_clusters) for clustered_words in img_clustered_words])
    X = img_bow_hist / np.sum(img_bow_hist) # normalise to account for different image shapes
    return X


def get_opencv_bow(data, num_clusters):
    bow = cv2.BOWKMeansTrainer(num_clusters)
    # input data of shape [NUM_IMAGES, BATCH_SIZE, NUM_DESCR, FEATURE_SIZE]
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            bow.add(data[i, 0, j, :])
    dictionary = bow.cluster()
        


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


                
def run3_SIFT_transforms(step_size):
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((255, 255)),
                               transforms.ToTensor(),
                               DenseSIFT(step_size=step_size)])

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