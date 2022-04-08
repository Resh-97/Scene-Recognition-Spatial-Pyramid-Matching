import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../CompVision")
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from CVcoursework.utils import create_dataset, run1_transforms

if __name__ == "__main__":
    transform = run1_transforms()
    train_dataset = create_dataset("./training/", transform, labeled=True)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # load training data 
    X_train = next(iter(train_dataloader))[0].numpy()
    y_train = next(iter(train_dataloader))[1].numpy()

    print(X_train.shape)
    # load in test data
    test_dataset = create_dataset("./testing/", transform, labeled=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    X_test = next(iter(test_dataloader))[0].numpy()
    # note that our y_test are UNLABELED
    y_test = next(iter(test_dataloader))[1].numpy()

    # plot validation accuracy for different values of K
    K_vals=[]
    accuracies=[]
    max_acc=0
    for K in range(1, 25):
        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(X_train, y_train)
        accuracies.append(clf.score(X_train, y_train))
        if accuracies[K-1]>max_acc:
            max_acc=accuracies[K-1]
            index=K-1
        K_vals.append(K)

    plt.figure()
    plt.plot(K_vals, accuracies)
    plt.show()
    print("max acc at k="+str(index+1)+" acc of "+str(max_acc))
