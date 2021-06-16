from hyperopt import hp

# print('Importing model spaces')

space_xgb ={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'learning_rate' : hp.uniform('learning_rate', 0.005, 0.5),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }



space_ada = {'learning_rate' : hp.uniform('learning_rate', 0.005, 1),
        'loss': hp.choice('loss', ['linear', 'square', 'exponential']),
        'seed': 0
    }

space_gbrt ={'loss': hp.choice('loss', ['ls', 'lad', 'huber', 'quantile']),
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'learning_rate' : hp.uniform('learning_rate', 0.005, 0.5),
        'seed': 0
    }

space_log ={'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
        'C': hp.uniform("C", 0.01, 1),
        'solver' : hp.choice('solver', ['liblinear', 'saga']),
        'seed': 0
    }

space_svm ={'C': hp.uniform("C", 0.01, 1),
        'kernel' : hp.choice('kernel', ['poly', 'rbf']),
        'degree' :hp.choice('degree', [2, 3, 4]),
        'seed': 0
    }