# this file consists of ML algorithmns with non-exhaustive hyperparameter spaces

from hyperopt import hp
from hyperopt.pyll.base import scope

ada_loss_functions = ['linear', 'square', 'exponential']
ada_space = {'learning_rate' : hp.uniform('learning_rate', 0.005, 1),
        'loss': hp.choice('loss', ada_loss_functions),
        'random_state': 0
    }


log_space ={'penalty': 'elasticnet',
        'C': hp.uniform("C", 0.01, 1),
        'solver' : 'saga',
        'l1_ratio' : hp.uniform('l1_ratio', 0,1),
        'max_iter' : 5000,
        'random_state': 0
    }


xgb_fitting_setting = {'verbose' : False}  # model tuning similar to others
xgb_space ={'max_depth' : scope.int(hp.uniformint("max_depth", 3, 18)),
        'gamma': hp.uniform('gamma', 1, 9),
        'learning_rate' : hp.uniform('learning_rate', 0.005, 0.5),
        'reg_alpha' : scope.int(hp.uniformint('reg_alpha', 40, 180)),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : scope.int(hp.uniformint('min_child_weight', 0, 10)),
        'n_estimators': 180,
        'seed': 0
    }


svm_kernels = ['poly', 'rbf']
svm_kernel_degrees = [2, 3, 4]
svm_space ={'C': hp.uniform("C", 0.005, 1),
        'kernel' : hp.choice('kernel', svm_kernels),
        'degree' :hp.choice('degree', svm_kernel_degrees),
        'random_state': 0
    }



# # further algorithmns which may have integer hyperparameter inputs, hp.uniformint('hyperparameter', lower_bound,
# # upper_bound) can be used. However, the Hyperopt function `fmin` returns hyperparameters as floats, which is why the
# # the list INT_KEYS may be required
# from hyperopt.pyll.base import scope  # https://github.com/hyperopt/hyperopt/issues/508 - I did not manage with this
INT_KEYS = []
INT_KEYS = ['max_depth', 'reg_alpha', 'min_child_weight']


# extra xgboost
