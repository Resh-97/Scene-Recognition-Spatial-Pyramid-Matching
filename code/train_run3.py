import numpy as np
import matplotlib.pyplot as plt
import json 
import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import (
    create_dataset, 
    show_misclassified, 
    get_percent_misclassified,
    plot_cm, 
    CodeBook)
from features import run3_SIFT_transforms, get_bovw
from classifiers import SupportVectorClassifer

# example arg "train.py --metrics=run1" -> metrics_file = "run1_metrics.txt"
#metrics_file = args.run + '_metrics.txt'
#output_file = 'run' + args.run + '.txt'


if __name__ == "__main__":
    seed=10
    # set random seed
    np.random.seed(seed)
    # dictionary to record run statistics
    stats = {}
    stats['seed'] = seed
    ##----------------------------------- READ DATASETS ----------------------------------------##
    feature_name = 'dense_sift'
    feature_hparams = {'step_size':3}

    transform = run3_SIFT_transforms(**feature_hparams)
    dataset = create_dataset("./training/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # add feature to stats
    stats['feature'] = {'name':feature_name, 'hparams': feature_hparams}

    # load labeled data 
    X, y, paths, path_class_idxs = next(iter(dataloader))
    X=X.numpy()
    y=y.numpy()

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(X,y, paths, test_size=0.2, stratify=y)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(X_train.squeeze(1).shape)
    ##-------------------------------------------------------------------------------------------##

    ##----------------------------------- BATCH FEATURE TRANSFORMS ----------------------------------------##
    tune_code_book_size = False
    code_book_cluster_num_candidates = [400]
    codebook = CodeBook()
    codebook.create_code_book(X_train.squeeze(1), tune_code_book_size, code_book_cluster_num_candidates)
    quantised_train_features = codebook.get_quantised_image_features(X_train.squeeze(1))
    quantised_val_features = codebook.get_quantised_image_features(X_val.squeeze(1))
    
    print("Quantisation done..")
    stats['bag_of_words'] = {'codebook_size':400}
    # bag of visual words
    """ num_clusters=700
    X = get_bovw(X, num_clusters=num_clusters)
    stats['bovw'] = {'num_clusters':num_clusters}"""
    # load in test data
    #test_dataset = create_dataset("./testing/", transform, labeled=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    # X_test = next(iter(test_dataloader))[0].numpy()
    # note that our y_test are UNLABELED
    #y_test = next(iter(test_dataloader))[1].numpy()


    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------ SAVE BOVW DATASET ------------------------------------##
    """
    with open('X_bovw.npy', 'wb') as f:
        np.save(f, X)
    with open('labels.npy') as f:
        np.save(f, y)
    """
    ##-------------------------------------------------------------------------------------------##


    ##------------------------------------- TUNE CLASSIFIER -------------------------------------##
    #params = config['params'] # READ FROM CONFIG FILE
    classifier_name = 'SVC'
    classifier_hparams = {'C':1, 'random_state':seed}
    # params for instantiating classifier
    # parameter grid to search over when tuning classifier
    param_grid = [
        {'C':[2**i for i in range(-5, 15)]}, 
        {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
        {'gamma':['scale', 'auto']},
        {'random_state':[seed]}
    ]
    classifier = SupportVectorClassifer(**classifier_hparams)
    best_score, best_params = classifier.tune(
        quantised_train_features,
        y_train, 
        param_grid=param_grid
        ) # tunes classifier and writes to metrics file
    print(f"Best parameters: {best_params}\n")
    print(f"Best score: {best_score}\n")

    # add classifier to stats dictionary
    stats['classifier'] = {'name':classifier_name, 'hparams':best_params}
    stats['val_accuracy'] = classifier.clf.score(quantised_val_features, y_val)



    ##-------------------------------------------------------------------------------------------##



    ##-------------------------------------- WRITE PREDICTIONS ----------------------------------##

    #classifier.predict(X_test, output_file)
    # evaluation
    class_labels_to_idx = dataset.class_to_idx
    preds = classifier.predict(X_val)  
    show_misclassified(preds, y_val, paths_val, class_labels_to_idx)
    percent_misclassified = get_percent_misclassified(preds, y_val, class_labels_to_idx)
    stats['percent_misclassified'] = percent_misclassified
    print(stats)
    # save stats
    with open('stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    # save classifier
    joblib.dump(classifier.clf, classifier_name+'.joblib')

    # get confusion matrix
    plot_cm(y_true=y_val, y_pred=preds, labels=dataset.classes)
    plt.show()
    #plt.figure()
   # plt.plot(K_vals,  avg_prec)
    #plt.show()
    #print("max acc at k="+str(index+1)+" avg prec of "+str(max_avg_prec))
    ##-------------------------------------------------------------------------------------------##
