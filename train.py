import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import create_dataset
from features import run1_transforms
from classifiers import KNearestNeighbors



# example arg "train.py --metrics=run1" -> metrics_file = "run1_metrics.txt"
#metrics_file = args.run + '_metrics.txt'
#output_file = 'run' + args.run + '.txt'


if __name__ == "__main__":
    # set random seed
    np.random.seed(10)
    ##----------------------------------- READ DATASETS ----------------------------------------##
    transform = run1_transforms(crop=200)
    dataset = create_dataset("./training/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # load labeled data 
    X = next(iter(dataloader))[0].numpy()
    y = next(iter(dataloader))[1].numpy()

    print(X.shape)
    # load in test data
    #test_dataset = create_dataset("./testing/", transform, labeled=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    # X_test = next(iter(test_dataloader))[0].numpy()
    # note that our y_test are UNLABELED
    #y_test = next(iter(test_dataloader))[1].numpy()

    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2, stratify=y)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------- TUNE CLASSIFIER -------------------------------------##
    #params = config['params'] # READ FROM CONFIG FILE
    # params for instantiating classifier
    params = {'n_neighbors':2, 'n_jobs':-1}
    # parameter grid to search over when tuning classifier
    param_grid = {
        'n_neighbors':[i for i in range(1, 25)],
        'n_jobs':[-1]
    }
    classifier = KNearestNeighbors(**params)
    best_score, best_params = classifier.tune(
        X_train,
        y_train, 
        param_grid=param_grid
        ) # tunes classifier and writes to metrics file
    print(f"Best parameters: {best_params}\n")
    print(f"Best score: {best_score}\n")

    ##-------------------------------------------------------------------------------------------##

    ##-------------------------------------- WRITE PREDICTIONS ----------------------------------##

    #classifier.predict(X_test, output_file)

    #plt.figure()
   # plt.plot(K_vals,  avg_prec)
    #plt.show()
    #print("max acc at k="+str(index+1)+" avg prec of "+str(max_avg_prec))
    ##-------------------------------------------------------------------------------------------##
