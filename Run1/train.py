import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.path.append("../../CompVision")
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from CVcoursework.utils import create_dataset, run1_transforms


def get_avg_precision(y_true, y_pred):
    precision=[]
    for idx, label in enumerate(np.unique(y_true)):
        precision.append(precision_score(y_true == label, y_pred == label))
    return np.mean(precision)


if __name__ == "__main__":
    np.random.seed(10)
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

    # plot validation accuracy for different values of K

    # let's split up the labeled training data into validation and testing
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.1, stratify=y)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    K_vals=[]
    accuracies=[]
    avg_prec = []
    max_acc=0
    max_avg_prec=0
    for K in range(1, 30):
        clf = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracies.append(clf.score(X_val, y_val))
        #avg_prec.append(np.mean(precision_score(y_val, clf.predict_proba(X_val), average=None)))
        avg_prec.append(get_avg_precision(y_val, y_pred))
        if accuracies[K-1]>max_acc:
            max_acc=accuracies[K-1]
            max_avg_prec=avg_prec[K-1]
            index=K-1
        K_vals.append(K)

    plt.figure()
    plt.plot(K_vals,  avg_prec)
    plt.show()
    print("max acc at k="+str(index+1)+" avg prec of "+str(max_avg_prec))
