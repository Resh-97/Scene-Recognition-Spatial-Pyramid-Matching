import numpy as np
from PIL import Image
import cv2
from torch import flatten
from torchvision import transforms


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

##------------------------------------------------------------------------------------------##

##-------------------------------------- TRANSFORM FUNCTIONS -------------------------------##
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

##-------------------------------------------------------------------------------------------##