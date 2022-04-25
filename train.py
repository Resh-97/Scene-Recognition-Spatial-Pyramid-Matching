import numpy as np
import os

from classifiers import LogisticRegression
from utils import imglist, create_dataset
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
    size_of_code_book = 500
    code_book, feature_scaler = create_code_book(X, size_of_code_book)
    quantised_features = get_quantised_img_features(X, size_of_code_book, code_book, feature_scaler)
    
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
    
    quantised_test_features = get_quantised_img_features(X_test, size_of_code_book, code_book, feature_scaler)
    
    y_preds = clf.predict(quantised_test_features)
    
##----------------------------------- TODO: WRITE TO FILE ----------------------------------------##