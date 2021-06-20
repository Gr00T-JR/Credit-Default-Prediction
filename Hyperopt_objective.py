import numpy as np
import pandas as pd

from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score


def build_objective_func(model, X_train : pd.DataFrame,y_train : pd.DataFrame, 
                         cross_val_method : int = 0, int_keys : list = [], **kwargs):
    """
    Builds objectives function that require `space` as input  from hyperopt `fmin`
    :param model: uninitialized ML model class with `.fit` and `.predict`
    :param X_train: data frame with training data features values
    :param y_train: data frame with training data labels
    :param cross_val_method: cross validation method, num_splits sklearn KFold
    :param int_keys: list with model keys that are integers
    :param kwargs: additional arguments for fitting model, especially for xgboost (e.g. evalmetrics)
    :return: objective function with that takes in input `space` during hyperopt `fmin`
    """

    # saved_kwargs = kwargs       # could I just use kwargs in the new function (?) - to be tested

    if cross_val_method == 0:
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25,
                                                                     random_state=0)

        def objective1(space):

            for key in int_keys:
                space[key] = int(space[key])

            if 'eval_set' in kwargs.keys():
                kwargs['eval_set'] = [(X_train, y_train), (X_validation, y_validation)]

            # defining `working_model` an instance of the model class
            working_model = model(**space)

            working_model.fit(X=X_train, y=y_train, **kwargs)
            pred = working_model.predict(X_validation)
            accuracy = accuracy_score(y_validation, pred > 0.5)
            print("SCORE:", accuracy)
            return {'loss': -accuracy, 'status': STATUS_OK}

        return objective1


    if cross_val_method > 0:
        kf = KFold(n_splits=cross_val_method, shuffle=True, random_state=252)
        KFold_list_train_indices = []
        KFold_list_validation_indices = []

        for train_indices, validation_indices in kf.split(X_train.to_numpy()):
            KFold_list_train_indices.append(train_indices)
            KFold_list_validation_indices.append(validation_indices)

        zipped = zip(KFold_list_train_indices, KFold_list_validation_indices)

        def objective2(space : dict) -> dict:

            for key in int_keys:
                space[key] = int(space[key])

            working_model = model(**space)

            scores = []
            for train_indices, validation_indices in zipped:

                if 'eval_set' in kwargs.keys():
                    kwargs['eval_set'] = [(X_train, y_train), (X_validation, y_validation)]

                working_model.fit(X_train[train_indices], y_train[train_indices], **kwargs)
                validation_predicted = working_model.predict(X_train[validation_indices])
                accuracy = accuracy_score(y_train[train_indices], validation_predicted > 0.5)
                scores.append(accuracy)

            print("SCORE:", np.mean(scores))
            return {'loss': -np.mean(scores), 'status': STATUS_OK}

        return objective2
