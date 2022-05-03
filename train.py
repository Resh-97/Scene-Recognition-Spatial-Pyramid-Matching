import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_dataset, avg_accuracy
from features import computeDenseSIFT, getDescriptors, clusterFeatures, getHistogramSPM, load_model
#from classifiers import KNearestNeighbors, LinearSVC
from sklearn.svm import LinearSVC


# example arg "train.py --metrics=run1" -> metrics_file = "run1_metrics.txt"
#metrics_file = args.run + '_metrics.txt'
#output_file = 'run' + args.run + '.txt'


if __name__ == "__main__":
    # set random seed
    np.random.seed(10)
    ##----------------------------------- READ DATASETS ----------------------------------------##
    train_data, train_label = load_dataset('./CVcoursework/images/training/')

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    trainX, valX, trainY, valY = train_test_split(train_data, train_label, train_size=0.8, random_state=42)

    ##-------------------------------------Extract Features-----------------------------------------##
    # Extract dense sift features from training images
    x_train = computeDenseSIFT(data = trainX)
    all_train_descriptors = getDescriptors(x_train)
    kmeans = clusterFeatures(all_train_descriptors, k=500)
    #kmeans = load_model('kmeansModel.sav')
    train_hist = getHistogramSPM(2, trainX, kmeans, 500)
    val_hist = getHistogramSPM(2, valX, kmeans, 500)
    ##------------------------------------- TUNE CLASSIFIER -------------------------------------##

    for c in np.arange(0.000307, 0.001, 0.0000462):
        clf = LinearSVC(random_state=0, C=c)
        clf.fit(train_hist, trainY)
        predict = clf.predict(val_hist)
        print ("C =", c, ",\t\t Accuracy:", np.mean(predict == valY)*100, "%")

    ##-------------------------------------------------------------------------------------------##

    ##-------------------------------------- WRITE PREDICTIONS ----------------------------------##

    #classifier.predict(X_test, output_file)

    #plt.figure()
   # plt.plot(K_vals,  avg_prec)
    #plt.show()
    #print("max acc at k="+str(index+1)+" avg prec of "+str(max_avg_prec))
    ##-------------------------------------------------------------------------------------------##
