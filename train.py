import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_dataset, avg_accuracy, write_preds_to_file, write_classification_report_to_file
from features import computeDenseSIFT, getDescriptors, clusterFeatures, getHistogramSPM, load_model
#from classifiers import KNearestNeighbors, LinearSVC
from classifiers import LogisticRegressionWrapper as LogisticRegression
from sklearn.svm import LinearSVC


# example arg "train.py --metrics=run1" -> metrics_file = "run1_metrics.txt"
#metrics_file = args.run + '_metrics.txt'
#output_file = 'run' + args.run + '.txt'


if __name__ == "__main__":
    # set random seed
    np.random.seed(10)
    ##----------------------------------- READ DATASETS ----------------------------------------##
    train_data, train_label, _, class_idx_to_label = load_dataset('./CVcoursework/images/training/')

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    trainX, valX, trainY, valY = train_test_split(train_data, train_label, train_size=0.8, random_state=42)

    ##-------------------------------------Extract Features-----------------------------------------##
    # Extract dense sift features from training images
    # x_train = computeDenseSIFT(data = trainX)
    # all_train_descriptors = getDescriptors(x_train)
    # kmeans = clusterFeatures(all_train_descriptors, k=500)
    k = 200
    kmeans = load_model('kmeansModel.sav')
    train_hist = getHistogramSPM(2, trainX, kmeans, k)
    val_hist = getHistogramSPM(2, valX, kmeans, k)
    ##------------------------------------- TUNE CLASSIFIER -------------------------------------##
    '''
    for c in np.arange(0.000307, 0.001, 0.0000462):
        clf = LinearSVC(random_state=0, C=c)
        clf.fit(train_hist, trainY)
        predict = clf.predict(val_hist)
        print ("C =", c, ",\t\t Accuracy:", np.mean(predict == valY)*100, "%")
    '''
        
    clf = LogisticRegression(multi_class='ovr', max_iter=100)
    clf.train(train_hist, trainY)
    print("Validation performance score: " + str(clf.score(val_hist, valY)))
    y_preds_val = clf.predict(val_hist)
    
    target_names = class_idx_to_label.keys()
    file_name = "run3_val_report"
    write_classification_report_to_file(file_name, y_preds_val, valY, target_names)

    ##-------------------------------------------------------------------------------------------##
    X_test, _, paths_test, _ = load_dataset('./CVcoursework/images/testing/', train_set=False)
    test_hist = getHistogramSPM(2, X_test, kmeans, k)
    y_preds = clf.predict(test_hist)
    
    print("Test run complete..")
    ##----------------------------------- WRITE PREDICTIONS TO FILE ----------------------------------------##

    file_name = 'run3'
    write_preds_to_file(file_name, paths_test, y_preds, class_idx_to_label)
    print("Predictions written to file " + file_name + ".txt")
