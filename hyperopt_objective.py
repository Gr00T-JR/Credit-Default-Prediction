import numpy as np
import random
# import pickle

from hyperopt import STATUS_OK
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score


def build_objective_func(algorithmn, X_train : np.ndarray,y_train : np.ndarray,
                         tuning_method : str, tuning_value,
                         tuning_measure = 'accuracy', random_state : int = 0,**fitting_setting):
    """
    Builds objectives function that require `space` as input during hyperopt `fmin`
    :param algorithmn: ML algorithmn (/ unitialized ML model)
    :param X_train: data frame with training data features values
    :param y_train: data frame with training data labels
    :param tuning_method: method for tuning hyerparameters, either 'train_validation_split' or 'KFold'
    :param tuning_value: tuning method value
    :param tuning_measure: measure of score/performance on the validation set, either 'roc' or 'accuracy'
    :param random_state: random_state
    :param kwargs: additional arguments for fitting model, especially for xgboost (e.g. evalmetrics)
    :return: objective function with that takes in input `space` during hyperopt `fmin`
    """
    possible_tuning_methods = ['train_validation_split', 'train_validation_split_randomized' 'KFold']

    # saved_kwargs = kwargs       # could I just use kwargs in the new function (?) - to be tested
    if tuning_measure == 'accuracy tuning':
        scoring_function = accuracy_score

    if tuning_measure == 'roc auc tuning':
        scoring_function = roc_auc_score

    if tuning_method == 'train_validation_split_randomized':

        def objective0(space):
            random_state = random.randint(0,1000)

            X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train, y_train, test_size=tuning_value,
                                                                            random_state=random_state)
            # defining `tuning_model` an instance of the model class
            tuning_model = algorithmn(**space)

            tuning_model.fit(X=X_train, y=y_train, **fitting_setting)
            pred = tuning_model.predict(X_validation_sub) > 0.5  # binary classification
            score = scoring_function(y_validation_sub, pred)
            print("SCORE:", score)
            return {'loss': -score, 'status': STATUS_OK}

        return objective0

    if tuning_method == 'train_validation_split':
        X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train, y_train, test_size=tuning_value,
                                                                     random_state=random_state)

        def objective1(space):
            # defining `tuning_model` an instance of the model class
            tuning_model = algorithmn(**space)

            tuning_model.fit(X=X_train_sub, y=y_train_sub, **fitting_setting)
            pred = tuning_model.predict(X_validation_sub) > 0.5  # binary classification
            score = scoring_function(y_validation_sub, pred)
            print("SCORE:", score)
            return {'loss': -score, 'status': STATUS_OK}

        return objective1


    if tuning_method == 'KFold':
        kf = KFold(n_splits=tuning_value, shuffle=True, random_state=random_state)
        KFold_list_train_indices = []
        KFold_list_validation_indices = []
        for train_indices, validation_indices in kf.split(X_train):
            KFold_list_train_indices.append(train_indices)
            KFold_list_validation_indices.append(validation_indices)

        def objective2(space: dict) -> dict:

            tuning_model = algorithmn(**space)
            scores = []

            for train_indices, validation_indices in zip(KFold_list_train_indices, KFold_list_validation_indices):
                X_train_temp, y_train_temp = X_train[train_indices], y_train[train_indices]
                X_validation_temp, y_validation_temp = X_train[validation_indices], y_train[validation_indices]
                tuning_model.fit(X_train_temp, y_train_temp, **fitting_setting)
                pred = tuning_model.predict(X_validation_temp) > 0.5

                scores.append(scoring_function(y_true=y_validation_temp, y_pred=pred))

            print("SCORE:", np.mean(scores))
            return {'loss': -np.mean(scores), 'status': STATUS_OK}

        return objective2

    if not tuning_measure in possible_tuning_methods:
        raise Exception(f'Tuning measure is not appropriate. Acceptable tuning methods are : {possible_tuning_methods}')
    if not (tuning_method == 'train_validation_split' or tuning_method == 'KFold'):
        raise Exception(f'Tuning method is not appropriate.')
