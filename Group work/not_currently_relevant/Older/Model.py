import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class Model:
    def __init__(self, model, X_train, y_train, X_test, y_test, scaler=StandardScaler()):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.y_lr = None
        self.report = None


    def update_data(self, X_train, y_train, X_test, y_test):
        # # # In case one wants to provide different split Option for data set or retrain on different set of features # # #
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __scale_data(self, data):
        return self.scaler.fit_transform(data)

    def get_name(self):
        return self.model
    
    def get_report(self):
        return self.report
    
    def train(self):
        X_train = self.__scale_data(self.X_train)   
        self.model.fit(X_train, self.y_train)
    
    def evaluate(self):
        X_test = self.__scale_data(self.X_test)
        self.y_lr = self.model.predict(X_test)
        self.report = classification_report(self.y_test, self.y_lr)

    def run(self):
        self.train()
        self.evaluate()
        return self.report