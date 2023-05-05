import os
import pandas as pd
from urllib.request import urlretrieve

from naive_bayes import NaiveBayes, pre_processing

DATASET_URL = os.getenv('DATASET_URL')
DATASET_PATH = os.getenv('DATASET_PATH', 'dataset.csv')


class ShiftPredictor:
    instance = None

    def __init__(self):
        if DATASET_URL:
            urlretrieve(DATASET_URL, DATASET_PATH)

        df = pd.read_csv(DATASET_PATH, keep_default_na=False)

        X, y = pre_processing(df)

        self.naive_bayes = NaiveBayes()
        self.naive_bayes.fit(X, y)

    @classmethod
    def get_instance(cls):
        if not ShiftPredictor.instance:
            ShiftPredictor.instance = ShiftPredictor()

        return ShiftPredictor.instance

    def predict(self, data):
        self.naive_bayes.predict([data])
        result = self.naive_bayes.results_probabilities[0]

        return result[0] == 'Yes', result[1]
