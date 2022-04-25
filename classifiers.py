from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class Classifier:
    def __init__(self, clf):
        self.clf = clf

    def train(self, X, y):
        self.clf.fit(X, y)

    def tune(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return self.clf.predict(X)



class LogisticRegression(Classifier):
    def __init__(self, **kwargs):
        self.scaler = MinMaxScaler()
        clf = LogisticRegression(**kwargs)
        super().__init__(clf)

    def tune(self, *args):
        pass
    
    def train(self, X, y):
        normalised_X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        
    def predict(self, X):
        normalised_X = self.scaler.transform(X)
        return self.clf.predict(normalised_X)
    
    def score(self, X, y):
        normalised_X = self.scaler.transform(X)
        return model.score(normalised_X, y)
