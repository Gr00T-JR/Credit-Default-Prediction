from hyperopt import STATUS_OK

import xgboost as xgb
import pickle
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

with open('objs.pkl', 'rb') as f:
    X_train_, y_train_, X_validation, y_validation = pickle.load(f)
f.close()

# print('importing')

def objective_xgb(space):

    model = xgb.XGBRegressor(
        max_depth = int(space['max_depth']),
        gamma = space['gamma'],
        learning_rate=space['learning_rate'],
        reg_alpha = int(space['reg_alpha']),
        reg_lambda=space['reg_lambda'],
        colsample_bytree=space['colsample_bytree'],
        min_child_weight=space['min_child_weight'],
        n_estimators=int(space['n_estimators']),
    )

    evaluation = [( X_train_, y_train_), ( X_validation, y_validation)]

    model.fit(X_train_, y_train_,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)


    pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

def objective_ada(space):
    model = AdaBoostRegressor(
        # algorithm = space['algorithm'],
        learning_rate=space['learning_rate'],
        n_estimators=int(space['n_estimators']),
    )

    model.fit(X_train_, y_train_)

    pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

def objective_gbrt(space):

    model = GradientBoostingRegressor(
        max_depth = space['max_depth'],
        learning_rate=space['learning_rate'] ,
        loss = space['loss'],
        n_estimators=int(space['n_estimators']),
    )

    # evaluation = [( X_train_, y_train_), ( X_validation, y_validation)]

    model.fit(X_train_, y_train_)


    pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


def objective_log(space):

    model = LogisticRegression(
        penalty= space['penalty'],
        C= space['C'],
        solver=space['solver'])

    try :
        model.fit(X_train_, y_train_)
    except: # in case the solver is unable to find an optimum
        return {'loss': -10e5, 'status': STATUS_OK }

    pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

def objective_svm(space):

    model = SVC(C= space['C'],
             kernel=space['kernel'],
             degree=space['degree'])


    model.fit(X_train_, y_train_)

    pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }