#import required libraries
from explore import wrangle #created module to clean data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

#load and split dataset
def load_data():
     data = wrangle('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
     X_train, X_test, y_train, y_test = train_test_split(data.drop('Churn', axis=1), 
                                                        data['Churn'],
                                                        test_size=.2,
                                                        stratify=data['Churn'],
                                                        random_state=42)
     return X_train, X_test, y_train, y_test

#load and train the catboost model
def train_model(X_train, y_train, X_test):
    model = CatBoostClassifier()
    model.load_model('./catboost_model.skops')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

#return performance metrics
def return_metrics(y_test, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred, pos_label=1),
        'Precision': precision_score(y_test, y_pred, pos_label=1),
        'Recall': recall_score(y_test, y_pred, pos_label=1),
        'f1_score': f1_score(y_test, y_pred, pos_label=1),
        'AUC': roc_auc_score(y_test, y_pred, pos_label=1)
    }
    print(pd.DataFrame(metrics))

def main():
    X_train, X_test, y_train, y_test = load_data()
    model, y_pred = train_model(X_train, y_train, X_test)

if __name__ == '__main__':
    main()