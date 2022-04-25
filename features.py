import numpy as np
from PIL import Image
import cv2
from torch import flatten, tensor
from torchvision import transforms


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
        x, y = minx, miny;
        hasNext = True

        while hasNext:
            nextX = x + step_x;
            nextY = y;
            if (nextX + window_width > maxx):
                nextX = minx;
                nextY += step_y;
            rec_dim = [x, y, window_width, window_height]

            rectangles.append(rec_dim);
            x = nextX;
            y = nextY;

            if (y + window_height > maxy):
                hasNext = False
        #print("All rectangular patches retrieved.......")
        return rectangles

class RemoveColourChannel:
    def __call__(self, img):
        img = img.squeeze()
        return img

##------------------------------------------------------------------------------------------##

def run2_transforms():
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((256, 256)),
                               transforms.ToTensor(),
                               RemoveColourChannel(),
                               ExtractFeatures()
                              ])

##-------------------------------------------------------------------------------------------##
