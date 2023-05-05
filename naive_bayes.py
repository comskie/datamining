import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
    """	score = (y_true - y_pred) / len(y_true) """

    return round(float(sum(y_pred == y_true)) / float(len(y_true)) * 100, 2)


def pre_processing(df: pd.DataFrame):
    """ partitioning data into features and target """

    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]

    return X, y


def train_test_split(x, y, test_size=0.25, random_state=None):
    """ partitioning the data into train and test sets """

    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, x_test, y_train, y_test


class NaiveBayes:
    def __init__(self):
        self.features = []
        self.class_priors = {}
        self.likelihoods = {}

        self.X_train = np.array([])
        self.y_train = np.array([])
        self.train_size = 0
        self.num_features = 0

        self.results_probabilities = []

    def fit(self, X: np.array, y: np.array):
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_features = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}
            # self.pred_priors[feature] = {}

            for feat_val in np.unique(self.X_train[feature]):
                # self.pred_priors[feature].update({feat_val: 0})

                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature].update({feat_val + '_' + outcome: 0})
                    self.class_priors.update({outcome: 0})

        self._calculate_class_priors()
        self._calculate_likelihoods()

        self.results_probabilities = []

    def predict(self, X):
        X = np.array(X)
        results = []
        self.results_probabilities = []

        for query in X:
            prob_outcome = {}

            likelihood_outcome = {}
            for outcome in np.unique(self.y_train):
                likelihood = np.product(
                    [self.likelihoods[feat][feat_val + '_' + outcome] for feat, feat_val in zip(self.features, query)])
                likelihood_outcome[outcome] = likelihood

            total = sum(likelihood_outcome.values())

            for outcome in np.unique(self.y_train):
                prob_outcome[outcome] = likelihood_outcome[outcome] / total

            # print(prob_outcome, sum(prob_outcome.values()))
            result = max(prob_outcome, key=lambda x: prob_outcome[x])
            self.results_probabilities.append([result, prob_outcome[result]])
            results.append(result)

        return np.array(results)

    def _calculate_class_priors(self):
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calculate_likelihoods(self):
        for feature in self.features:
            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()] \
                    .value_counts().to_dict()
                for feat_val, count in feat_likelihood.items():
                    self.likelihoods[feature][feat_val + '_' + outcome] = count / outcome_count


if __name__ == "__main__":
    # Weather Dataset
    print("\nCSIT Retention:")

    df = pd.read_csv("dataset.csv", keep_default_na=False)
    # print(df)

    # Split features and target
    X, y = pre_processing(df)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # print(X_train, y_train)
    nb_clf = NaiveBayes()
    nb_clf.fit(X_train, y_train)
    # print(X_train, y_train)

    print("Train Accuracy: {}".format(accuracy_score(y_train, nb_clf.predict(X_train))))
    print("Test Accuracy: {}".format(accuracy_score(y_test, nb_clf.predict(X_test))))

    # Query 1:
    query = np.array([['Male', 'BSIT', 'TVL', 'Yes', 'No', '84-80', 'Yes', '1-4', '0', 'Yes', 'Yes', 'Yes']])
    print("Query 1:- {} ---> {}".format(query, nb_clf.predict(query)))
    print('Probabilities:', nb_clf.results_probabilities)
#
