import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import create_dataset, show_misclassified, get_percent_misclassified, train_model
from features import run3_transforms
from classifiers import KNearestNeighbors
from torchvision.models import resnet18
from torch import nn, optim, from_numpy

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
    # labels (used to illustrate incorrectly classified images)
    class_labels=["bedroom", "Coast", "Forest", "Highway", "industrial", 
    "Insidecity", "kitchen", "livingroom", "Mountain", "Office", "OpenCountry", "store", 
    "Street", "Suburb", "TallBuilding"]
    ##----------------------------------- READ DATASETS ----------------------------------------##
    feature_name = 'no_features'
    feature_hparams = {'crop':200}

    transform = run3_transforms(**feature_hparams)
    dataset = create_dataset("./training/", transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # add feature to stats
    stats['feature'] = {'name':feature_name, 'hparams': feature_hparams}

    # load labeled data 
    X, y, paths, path_class_idxs = next(iter(dataloader))
    X=X.numpy()
    y=y.numpy()
    


    # load in test data
    #test_dataset = create_dataset("./testing/", transform, labeled=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    # X_test = next(iter(test_dataloader))[0].numpy()
    # note that our y_test are UNLABELED
    #y_test = next(iter(test_dataloader))[1].numpy()

    ##-------------------------------------------------------------------------------------------##

    ##------------------------------------ SPLIT DATASETS ---------------------------------------##
    # let's split up the labeled training data into validation and testing
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(X,y, paths, test_size=0.2, stratify=y)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    ##-------------------------------------------------------------------------------------------##


    #----------------------------------- Deep Learning ---------------------------------#
    data = {
        "train": [from_numpy(X_train).unsqueeze(0), from_numpy(y_train).unsqueeze(0)],
        "val": [from_numpy(X_val).unsqueeze(0), from_numpy(y_val).unsqueeze(0)]
    }
    model_conv = resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 15)

    model_conv = model_conv.to('cpu')

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    finetuned_model = train_model(model_conv, 
                                data,
                                criterion, 
                                optimizer_conv, 
                                exp_lr_scheduler, 
                                num_epochs=25)
    #----------------------------END Deep Learning ---------------------------------------#



    ##-------------------------------------- WRITE PREDICTIONS ----------------------------------##

    #classifier.predict(X_test, output_file)
    # evaluation
    class_labels_to_idx = dataset.class_to_idx
    preds = finetuned_model(from_numpy(X_val)).numpy()
    #preds = classifier.predict(X_val)  UNCOMMENT WHEN DEEP LEARNING DONE
    show_misclassified(preds, y_val, paths_val, class_labels_to_idx)
    percent_misclassified = get_percent_misclassified(preds, y_val, class_labels_to_idx)
    stats['percent_misclassified'] = percent_misclassified
    print(stats)

    #plt.figure()
   # plt.plot(K_vals,  avg_prec)
    #plt.show()
    #print("max acc at k="+str(index+1)+" avg prec of "+str(max_avg_prec))
    ##-------------------------------------------------------------------------------------------##
