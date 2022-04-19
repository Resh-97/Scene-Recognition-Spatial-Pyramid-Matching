from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils import avg_precision


class Classifier:
    def __init__(self, clf):
        self.clf = clf

    def train(self, X, y):
        self.clf.fit(X, y)

    def tune(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return self.clf.predict(X)



class KNearestNeighbors(Classifier):
    def __init__(self, **kwargs) -> None:
        # define classifier
        clf = KNeighborsClassifier(**kwargs)
        # init superclass
        super().__init__(clf)

    def tune(self, X, y, param_grid):
        """
        Runs grid search over a parameter grid for 
        chosen classifier. 
        NOTE:
            Only works with SKlearn classifiers 
            (as far as I know). 
            Custom implementation may need to overwrite.

        Args:
            X (numpy.array): training data.
            y (numpy.array): training labels.
            param_grid (dict of string:list): grid of parameters to search over.
        Returns:
            (float, dict of string:float) best score and best parameters.
        """
        search = GridSearchCV(
            self.clf, 
            param_grid, 
            scoring=avg_precision,
            n_jobs=-1
            ).fit(X, y)
        best_params = search.cv_results_['params'][search.best_index_]
        best_score = search.best_score_
        best_estimator = search.best_estimator_

        # change classifier to best estimator
        self.clf = best_estimator
        return best_score, best_params

        


