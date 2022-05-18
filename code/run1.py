import json
import random
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import (
    create_dataset, 
    show_misclassified, 
    get_percent_misclassified, 
    plot_cm, 
    write_preds_to_file)
from features import run1_transforms
from classifiers import KNearestNeighbors


if __name__ == "__main__":
    seed=10
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # dictionary to record run statistics
    stats = {}
    stats['seed'] = seed
    ##----------------------------------- READ DATASETS ----------------------------------------##
    feature_name = 'tiny_image'
    feature_hparams = {'crop':200}

    transform = run1_transforms(**feature_hparams)
    dataset = create_dataset("./training/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # add feature to stats
    stats['feature'] = {'name':feature_name, 'hparams': feature_hparams}

    # load labeled data 
    X, y, paths, path_class_idxs = next(iter(dataloader))
    X=X.numpy().astype(np.float32)
    y=y.numpy().astype(np.float32)
    

    # load in test data
    test_dataset = create_dataset("./testing/", transform, labeled=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    X_test, y_test, test_paths, test_path_class_idxs  = next(iter(test_dataloader))
    X_test=X_test.numpy().astype(np.float32)
    y_test=y_test.numpy().astype(np.float32)

    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(X,y, paths, test_size=0.2, stratify=y)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------- TUNE CLASSIFIER -------------------------------------##
    #params = config['params'] # READ FROM CONFIG FILE
    classifier_name = 'KNN'
    classifier_hparams = {'n_neighbors':2, 'n_jobs':-1}
    # params for instantiating classifier
    # parameter grid to search over when tuning classifier
    param_grid = {
        'n_neighbors':[i for i in range(1, 25)],
        'n_jobs':[-1]
    }
    classifier = KNearestNeighbors(**classifier_hparams)
    best_score, best_params = classifier.tune(
        X_train,
        y_train, 
        param_grid=param_grid
        ) # tunes classifier and writes to metrics file
    print(f"Best parameters: {best_params}\n")
    print(f"Best score: {best_score}\n")

    # add classifier to stats dictionary
    stats['classifier'] = {'name':classifier_name, 'hparams':best_params}


    ##-------------------------------------------------------------------------------------------##



    ##-------------------------------------- Evaluate classifier ----------------------------------##

    #classifier.predict(X_test, output_file)
    # evaluation
    stats['val_accuracy'] = classifier.clf.score(X_val, y_val)
    class_labels_to_idx = dataset.class_to_idx
    preds = classifier.predict(X_val)  
    show_misclassified(preds, y_val, paths_val, class_labels_to_idx)
    # get confusion matrix
    plot_cm(y_true=y_val, y_pred=preds, labels=dataset.classes)
    plt.show()
    percent_misclassified = get_percent_misclassified(preds, y_val, class_labels_to_idx)
    stats['percent_misclassified'] = percent_misclassified
    print(stats)
    with open('stats_run1.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    # save classifier
    joblib.dump(classifier.clf, classifier_name+'.joblib')

    ##-------------------------------------- Write predictions on test set -------------------------##

    #test_preds = classifier.predict(X_test)
    #file_name = 'run1'
    #write_preds_to_file(file_name, test_paths, test_preds, dataset.class_to_idx)
    #print("Predictions written to file " + file_name + ".txt")
    ##-------------------------------------------------------------------------------------------##
