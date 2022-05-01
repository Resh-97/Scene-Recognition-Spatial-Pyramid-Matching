import numpy as np
import os

from classifiers import LogisticRegressionWrapper as LogisticRegression
from utils import imglist, create_dataset, CodeBook
from features import run2_transforms
from torch.utils.data import DataLoader


# example arg "train.py --metrics=run1" -> metrics_file = "run1_metrics.txt"
#metrics_file = args.run + '_metrics.txt'
#output_file = 'run' + args.run + '.txt'


if __name__ == "__main__":
    # set random seed
    np.random.seed(10)
    
    ##----------------------------------- READ DATASETS ----------------------------------------##
    transform = run2_transforms()
    
    dataset = create_dataset("./training/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    X, y, paths, path_class_idxs = next(iter(dataloader))
    X = X.numpy()
    y = y.numpy()
    
##----------------------------------- BATCH FEATURE TRANSFORMS ----------------------------------------##
    size_of_code_book = 600
    codebook = CodeBook(size_of_code_book)
    codebook.create_code_book(X)
    quantised_features = codebook.get_quantised_image_features(X)
    
    print("Quantisation done..")
    
##----------------------------------- MODEL FITTING ----------------------------------------##
    
    clf = LogisticRegression(multi_class='ovr', max_iter=100)
    clf.train(quantised_features, y)
    
    print(clf.score(quantised_features, y))
    
##----------------------------------- TODO: VALIDATION ----------------------------------------##
    
##----------------------------------- TEST: READ DATASETS ----------------------------------------##
    
    dataset = create_dataset("./testing/", transform, labeled=False)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    X_test, _, _, _ = next(iter(dataloader))
    X_test = X_test.numpy()
    
    quantised_test_features = codebook.get_quantised_image_features(X_test)
    
    y_preds = clf.predict(quantised_test_features)
    print(y_preds[:10])
    
##----------------------------------- TODO: WRITE TO FILE ----------------------------------------##