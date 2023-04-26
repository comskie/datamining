import os
import numpy as np
import pandas as pd
from enum import Enum

from naive_bayes import NaiveBayes, pre_processing, train_test_split


class ShiftPredictor:
    instance = None

    def __init__(self):
        df = pd.read_csv("dataset.csv", keep_default_na=False)

        X, y = pre_processing(df)

        self.naive_bayes = NaiveBayes()
        self.naive_bayes.fit(X, y)

    @classmethod
    def get_predictor(cls):
        if not ShiftPredictor.instance:
            ShiftPredictor.instance = ShiftPredictor()

        return ShiftPredictor.instance

    def predict(self, data):
        self.naive_bayes.predict([data])
        return self.naive_bayes.results_probabilities[0]
