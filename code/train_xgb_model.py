import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score, 
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

#   PATHS  #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
STUDY_PATH = os.path.join(DATA_DIR, 'optuna_xgb_study.pkl')
MODEL_PATH = os.path.join(DATA_DIR, 'XGBoost75Accuracy')

train_path = os.path.join(DATA_DIR, 'train_preprocessed.csv')
test_path = os.path.join(DATA_DIR, 'test_preprocessed.csv')

#  DATA  #

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=['Credit_Score'])
y = train['Credit_Score']

X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# OPTUNA IMPORTS #

study = joblib.load(STUDY_PATH)
best_params = study.best_params


#  TRAINING #

mlflow.set_experiment('CreditScore_XGBoost')

with mlflow.start_run(run_name='XBG_Optuna_Tuned'):
    
    mlflow.log_params(best_params)

    model = XGBClassifier(
        **best_params,
        eval_metric='mlogloss',
        verbosity=1,
        objective='multi:softprob'
    )

    model.fit(
        X_subtrain, y_subtrain,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    prec = precision_score(y_val, y_pred, average='weighted')
    rec = recall_score(y_val, y_pred, average='weighted')

    print(f'\nAccuracy: {acc:.4f}')
    print('\nClassification report:\n', classification_report(y_val, y_pred))
    
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('precission', prec)
    mlflow.log_metric('recall', rec)
    
    mlflow.xgboost.log_model(
        model, 
        name='xgb_model',
        input_example=X_subtrain.iloc[:1],
        signature=mlflow.models.infer_signature(X_subtrain, y_pred[:1])
        )

joblib.dump(model, MODEL_PATH)
