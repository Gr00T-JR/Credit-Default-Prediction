# This function evaluates the model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def showConfusionMatrix(y_true ,y_predicted):
    """
    returns confusion matrix with the true y values and predicted y values
    """
    cm = confusion_matrix(y_true ,y_predicted)
    ConfusionMatrixDisplay(cm).plot()

def evalMetrics(y_true,y_predicted, confusion_matrix=False):

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
        showConfusionMatrix(y_true, y_predicted)

    return accuracy_score


def fitModel(tuned_model, X_fitting, y_fitting, eval_metrics=False, confusion_matrix = False):
    tuned_model.fit(X_fitting ,y_fitting)
    print(f'Performance on fitting data of tuned_model {tuned_model} \n')
    if eval_metrics or confusion_matrix:
        y_fitting_pred = tuned_model.predict(X_fitting ) >0.5
        if eval_metrics:
            evalMetrics(y_fitting, y_fitting_pred)
        if confusion_matrix:
            print(f'Confusion Matrix on the fitting data')
            showConfusionMatrix(y_fitting, y_fitting_pred)

    return tuned_model

def modelFitEvaluation(model, X_fitting, y_fitting, eval_metrics=False, confusion_matrix = False):
    model.fit(X_fitting ,y_fitting)
    print(f'Performance on fitting data of model {model} \n')
    if eval_metrics or confusion_matrix:
        y_fitting_pred = model.predict(X_fitting ) >0.5
        if eval_metrics:
            EvalMetrics(y_fitting, y_fitting_pred)
        if confusion_matrix:
            print(f'Confusion Matrix on the fitting data')
            ShowConfusionMatrix(y_fitting, y_fitting_pred)

    return model




