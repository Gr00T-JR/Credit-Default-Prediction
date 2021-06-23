import numpy as np
import pandas as pd

from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score


def build_objective_func(model, X_train : np.ndarray,y_train : pd.Series, tuning_measure = 'accuracy',
                         cross_val_method : int = 0, random_state : int = 0, **kwargs):
    """
    Builds objectives function that require `space` as input during hyperopt `fmin`
    :param model: uninitialized ML model class with `.fit` and `.predict`
    :param X_train: data frame with training data features values
    :param y_train: data frame with training data labels
    :param cross_val_method: cross validation method, num_splits sklearn KFold
    :param tuning_measure: measure of score/performance on the validation set, either 'roc' or 'accuracy'
    :param random_state: random_state
    :param kwargs: additional arguments for fitting model, especially for xgboost (e.g. evalmetrics)
    :return: objective function with that takes in input `space` during hyperopt `fmin`
    """

    # saved_kwargs = kwargs       # could I just use kwargs in the new function (?) - to be tested

    if cross_val_method == 0:
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25,
                                                                     random_state=random_state)

        def objective1(space):

            # defining `working_model` an instance of the model class
            working_model = model(**space)

            working_model.fit(X=X_train, y=y_train, **kwargs)
            pred = working_model.predict(X_validation) > 0.5  # binary classification
            if tuning_measure == 'accuracy':
                score = accuracy_score(y_validation, pred)
            if tuning_measure == 'roc':
                score = roc_auc_score(y_validation, pred)
            print("SCORE:", score)
            return {'loss': -score, 'status': STATUS_OK}

        return objective1


    if cross_val_method > 0:
        y_train = y_train.to_numpy()
        kf = KFold(n_splits=cross_val_method, shuffle=True, random_state=random_state)
        KFold_list_train_indices = []
        KFold_list_validation_indices = []

        for train_indices, validation_indices in kf.split(X_train):
            KFold_list_train_indices.append(train_indices)
            KFold_list_validation_indices.append(validation_indices)


        def objective2(space : dict) -> dict:

            working_model = model(**space)
            scores = []

            for train_indices, validation_indices in zip(KFold_list_train_indices, KFold_list_validation_indices):
                X_train_temp, y_train_temp = X_train[train_indices], y_train[train_indices]
                X_validation_temp, y_validation_temp = X_train[validation_indices], y_train[validation_indices]
                working_model.fit(X_train_temp, y_train_temp, **kwargs)
                pred = working_model.predict(X_validation_temp) > 0.5

                if tuning_measure == 'accuracy':
                    scores.append(accuracy_score(y_validation_temp, pred))
                if tuning_measure == 'roc':
                    scores.append(roc_auc_score(y_validation_temp, pred))

            print("SCORE:", np.mean(scores))
            return {'loss': -np.mean(scores), 'status': STATUS_OK}

        return objective2


# for key in int_keys:
#     space[key] = int(space[key])