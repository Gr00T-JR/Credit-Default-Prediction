import pandas as pd
from Model import Model
from sklearn.linear_model import LogisticRegression

def pd_load_data(dirName="DataFiles/"):
    dirName_train = dirName + 'CreditCard_train.csv'
    dirName_test = dirName + 'CreditCard_test.csv'
    return pd.read_csv(dirName_train), pd.read_csv(dirName_test) 

training_df, test_df = pd_load_data()
X_train, X_test = training_df.iloc[1:,1:-1], test_df.iloc[1:,1:-1]
y_train, y_test = training_df.loc[1:,'Y'], test_df.loc[1:,'Y']

model = Model(LogisticRegression(), X_train, y_train, X_test, y_test)
model_report = model.run()

#Equivalently You can run:
'''
#If you want to input different set of attributes for training
mode.update_data(X_train, y_train, X_test, y_test)

#Retrain Model
model.train()

#Run Evaluation function again
model.evaluate()

#Get the latest report of metrics
model_report = model.get_report()
'''

print(model_report)
