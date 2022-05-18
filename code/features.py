import numpy as np
import math
import cv2
import pickle
from PIL import Image
from torch import flatten, from_numpy, tensor
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans, KMeans


##-------------------------------------- FEATURE EXTRACTION CLASSES -----------------------## 
class ExtractFeatures:
    def __call__(self, img):
        rectangles = self.getAllRectangleKeyPointsForImg(img)
        features = []
        for x,y,w,h in rectangles:
            patch_img = img[x:x+w,y:y+w]
            img_array = np.array(patch_img)
            flat_arr = img_array.ravel()
            vector = flat_arr.tolist()
            features.append(vector)
        # send tensor
        return tensor(features)
    
    def getAllRectangleKeyPointsForImg(self, img, step=4, window=8): 
        minx, miny = 0, 0
        maxx, maxy = img.size(0), img.size(1)
        step_x, step_y = step, step
        window_width, window_height = window, window
        rectangles = [] 
        x, y = minx, miny
        hasNext = True

        while hasNext:
            nextX = x + step_x
            nextY = y
            if (nextX + window_width > maxx):
                nextX = minx
                nextY += step_y
            rec_dim = [x, y, window_width, window_height]

            rectangles.append(rec_dim)
            x = nextX
            y = nextY

            if (y + window_height > maxy):
                hasNext = False
        #print("All rectangular patches retrieved.......")
        return rectangles

class RemoveColourChannel:
    def __call__(self, img):
        img = img.squeeze()
        return img


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
        std = np.std(img)
        return Image.fromarray((x - mean) / std) 

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
        step_size = 2
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


def run2_transforms():
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((256, 256)),
                               transforms.ToTensor(),
                               RemoveColourChannel(),
                               ExtractFeatures()
                              ])


                
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