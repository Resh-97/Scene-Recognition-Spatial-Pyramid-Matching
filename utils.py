import os
import shutil
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torch import flatten
from torchvision import datasets, transforms


class ZeroMeanTransform:
    def __call__(self, img):
        x = np.array(img, dtype=np.float32)
        mean = np.mean(img)
        return Image.fromarray((x - mean))

class UnitLenTransform:
    def __call__(self, img):
        x = np.array(img)
        cv2.normalize(x, x, alpha=1, dtype=cv2.CV_32F)
        return Image.fromarray(x)


def create_dataset(dir, transform=None, labeled=True):
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

def run1_transforms(resize=16, crop=240):
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(255),
                               transforms.CenterCrop(crop),
                               transforms.Resize(resize),
                               ZeroMeanTransform(),
                               UnitLenTransform(),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: flatten(x))])



if __name__ == "__main__":
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize(255),
                                 transforms.CenterCrop(120),
                                 transforms.Resize(16),
                                 ZeroMeanTransform(),
                                 UnitLenTransform(),
                                 transforms.Lambda(lambda x: flatten(x)),
                                 transforms.ToTensor()])

    #data = BaseDataset('./testing/')
    #dataset = datasets.ImageFolder('./training/', transform)
    dataset = create_dataset("./testing/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(dataloader)))