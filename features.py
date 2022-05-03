import numpy as np
from PIL import Image
import cv2
import math
from torch import flatten
from torchvision import transforms
import pickle
from sklearn.cluster import KMeans
##-------------------------------------- FEATURE EXTRACTION CLASSES -----------------------##
def computeDenseSIFT(data=None,img=None):
    step_size = 4
    if data is not None:
        x = []
        print("SIFT descriptors computation began.....")
        for i in range(0, len(data)):
            sift = cv2.SIFT_create()
            image = data[i]

            keypoint = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size) for y in range(0, image.shape[1], step_size)]
            dense_features = sift.compute(image, keypoint)
            x.append(dense_features[1])
        print("SIFT descriptors computed..........")
        return x
    else:
        sift = cv2.SIFT_create()
        step_size = step_size
        keypoints = [cv2.KeyPoint(x, y, step_size)
            for y in range(0, img.shape[0], step_size)
                for x in range(0, img.shape[1], step_size)]

        descriptors = sift.compute(img, keypoints)[1]
        return descriptors

def getDescriptors(x_train):
    all_train_descriptors = []
    print("SIFT descriptors being appended.....")
    for i in range(len(x_train)):
        for j in range(x_train[i].shape[0]):
            all_train_descriptors.append(x_train[i][j,:])

    all_train_descriptors = np.array(all_train_descriptors)
    return all_train_descriptors

# form histogram with Spatial Pyramid Matching upto level L
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):
                desc = computeDenseSIFT(img = img[y:y+h_step, x:x+w_step])
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step

    hist = np.array(h).ravel()
    # normalize hist
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist


# get histogram representation for training data
def getHistogramSPM(level, data, kmeans, k):
    x = []
    for i in range(len(data)):
        hist = getImageFeaturesSPM(level, data[i], kmeans, k)
        x.append(hist)
    return np.array(x)

# build BoW presentation from SIFT of training images
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0, verbose=100).fit(all_train_desc)
    filename = 'kmeansModel3.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    return kmeans

# load the clustering model from disk
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

##-------------------------------------- TRANSFORM FUNCTIONS -------------------------------##
def run_transforms(resize=16, crop=240):
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
                               transforms.Resize((255,255)),
                               #transforms.CenterCrop(crop),
                               #transforms.Resize(resize),
                               ZeroMeanTransform(),
                               UnitLenTransform(),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: flatten(x))
                              ])

##-------------------------------------------------------------------------------------------##
