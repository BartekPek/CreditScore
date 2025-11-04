import os
import pandas as pd
import numpy as np
import optuna
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STUDY_PATH = "data/optuna_xgb_study.pkl"

train_path = os.path.join(DATA_DIR, 'train_preprocessed.csv')
test_path = os.path.join(DATA_DIR, 'test_preprocessed.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=['Credit_Score'])
y = train['Credit_Score']



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'objective': 'multi:softprob',   
        'num_class': 3,                  
        'tree_method': 'hist',           
        'random_state': 42,
        'verbosity': 0
    }
    
    model = XGBClassifier(**params, n_jobs=-1)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return np.mean(scores)

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20, show_progress_bar=True)

print('Best params: ', study.best_params)
print('Best Accuracy score: ', study.best_value)

joblib.dump(study, STUDY_PATH)

