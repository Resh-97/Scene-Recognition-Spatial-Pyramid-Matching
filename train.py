import numpy as np
import os

from classifiers import LogisticRegressionWrapper as LogisticRegression
from sklearn.model_selection import train_test_split
from utils import CodeBook, imglist, create_dataset, write_preds_to_file, write_classification_report_to_file
from features import run2_transforms
from torch.utils.data import DataLoader


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
        
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(X, y, paths, test_size=0.2, stratify=y)
    
        
    ##----------------------------------- BATCH FEATURE TRANSFORMS ----------------------------------------##
    tune_code_book_size = True
    code_book_cluster_num_candidates = [50, 200, 400, 500, 550, 600, 650]
    codebook = CodeBook()
    codebook.create_code_book(X_train, tune_code_book_size, code_book_cluster_num_candidates)
    quantised_train_features = codebook.get_quantised_image_features(X_train)
    quantised_val_features = codebook.get_quantised_image_features(X_val)
    
    print("Quantisation done..")
    
    ##------------------------------------------- MODEL FITTING -------------------------------------------##
    
    clf = LogisticRegression(multi_class='ovr', max_iter=100)
    clf.train(quantised_train_features, y_train)
    
    print("Model fitting done..")
        
    ##----------------------------------- VALIDATION DATASET SCORE ----------------------------------------##
    print("Validation performance score: " + str(clf.score(quantised_val_features, y_val)))
    y_preds_val = clf.predict(quantised_val_features)
    
    ##-------------------------------- WRITE CLASSIFICATION REPORT TO FILE --------------------------------## 
    class_to_label_dict = dataset.class_to_idx        
    
    target_names = class_to_label_dict.keys()
    file_name = "run2_val_report"
    write_classification_report_to_file(file_name, y_preds_val, y_val, target_names)
    
    print("Validation report saved to file: " + str(file_name) + ".txt")
    
    ##-------------------------------------------- TEST RUN -----------------------------------------------##
    
    dataset = create_dataset("./testing/", transform, labeled=False)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    X_test, _, paths_test, _ = next(iter(dataloader))
    X_test = X_test.numpy()
    
    quantised_test_features = codebook.get_quantised_image_features(X_test)
    y_preds = clf.predict(quantised_test_features)
    
    print("Test run complete..")
        
    ##----------------------------------- WRITE PREDICTIONS TO FILE ----------------------------------------##

    file_name = 'run2'
    write_preds_to_file(file_name, paths_test, y_preds, class_to_label_dict)
    print("Predictions written to file " + file_name + ".txt")