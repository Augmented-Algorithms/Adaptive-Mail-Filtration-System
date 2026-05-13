from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class TrainModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}  

    def logistic_regression(self):
        self._train_and_score("Logistic Regression", LogisticRegression())

    def random_forest(self):
        self._train_and_score("Random Forest", RandomForestClassifier())

    def svm(self):
        self._train_and_score("SVM", SVC())

    def _train_and_score(self, name, model):
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, pred) * 100
        self.results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}%")

    def compare_all(self):
        self.logistic_regression()
        self.random_forest()
        self.svm()
        print("\nBest Model: ")
        best = max(self.results, key=self.results.get)
        print(f"{best}: {self.results[best]:.2f}%")