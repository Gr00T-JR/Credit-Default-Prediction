# This function evaluates the model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def show_confusion_matrix(y_true ,y_predicted):
    """
    returns confusion matrix with the true y values and predicted y values
    """
    cm = confusion_matrix(y_true ,y_predicted)
    ConfusionMatrixDisplay(cm).plot()

def performance_metrics(y_true,y_predicted, confusion_matrix=False):

    # metrics used here are: Accuracy, Recall, Precision, ROC/AUC and F1.
    # these are the industry standard and provide a proper, unbiased benchmark for models.

    accuracy_score = metrics.accuracy_score(y_true, y_predicted)
    recall_score = metrics.recall_score(y_true, y_predicted)
    precision_score = metrics.precision_score(y_true, y_predicted)
    roc_auc_score = metrics.roc_auc_score(y_true, y_predicted)
    f1_score = metrics.f1_score(y_true, y_predicted)

    print("Accuracy score: " + accuracy_score.astype(str))
    print("Recall score: " + recall_score.astype(str))
    print("Precision_score: " + precision_score.astype(str))
    print("ROC/AUC score: " + roc_auc_score.astype(str))
    print("F1 score: " + f1_score.astype(str))
    print("\n")

    if confusion_matrix:
        print(f'Confusion Matrix with the true and predicted data')
        show_confusion_matrix(y_true, y_predicted)

    return accuracy_score


# def fit_model(model, X_fitting, y_fitting, verbose = False, eval_metrics=False, confusion_matrix = False):
#     fitted_model = model.fit(X_fitting ,y_fitting)
#     if verbose:
#         print(f'Performance on fitting data of fitted_model {fitted_model} \n')
#         if eval_metrics or confusion_matrix:
#             y_fitting_pred = fitted_model.predict(X_fitting ) >0.5
#             if eval_metrics:
#                 eval_metrics(y_fitting, y_fitting_pred)
#             if confusion_matrix:
#                 print(f'Confusion Matrix on the fitting data')
#                 show_confusion_matrix(y_fitting, y_fitting_pred)
#
#     return fitted_model





