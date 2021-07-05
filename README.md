# Credit-Default-Prediction

__Abstract:__ This project considers how KFold and a fixed train-validation affects the performance of the scikit-learn
estimator *AdaBoostRegressor* on a credit card default dataset. The dataset consists of 30,000 labelled points with 23
features. The Hyperopt optimization framework is used with 30 search iterations for hyperparameters per optimization.
Ten simulations found model tuning using KFold is higher than a fixed train-validation, set at a p-value
of 0.0134. The respective average test accuracy is 83.355% and 83.42%. At a confidence level of 95%, KFold tuning is has  increases the performance of test accuracy 0.0224%, given
the simulation settings. Further performance measures, such as roc-auc score and real good of probability estimation
(found using the smooth sorting method), were not found to have difference at 95% confidence level.

* __[Link to original paper](https://www.sciencedirect.com/science/article/pii/S0957417407006719)__

* __[Link to UCI dataset page](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)__
<br>

### Project Method
1. data preparation, including imports and scaling
2. hyperparameter optimization space
3. objective function build
4. model build using Hyperopt
5. model evaluation, includes 'smooth sorting method'
