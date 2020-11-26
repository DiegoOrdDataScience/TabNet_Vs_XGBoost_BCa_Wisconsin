import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
import os
import matplotlib.pyplot as plt

# For reproducible results
np.random.seed(0)


# Load the the csv file (BCa Wisconsin Diagnostic Data Set (Kaggle).csv)
BCa_address = '/Users/francybayona/Documents/DOC DIEGO/Python docs/Nanostic Project/BCa Wisconsin Diagnostic Data Set (Kaggle).csv'
df_BCa = pd.read_csv(BCa_address)
df_BCa.head(1)

#Define train and target for TabNet
#Drop columns that are not part of the analysis: Unnamed: 32 and id
train = df_BCa.drop(['id', 'Unnamed: 32'],axis='columns')
target = 'diagnosis'

#Define the number of folds for the outer cross-validation
outer_cv_folds = 5

# Initialize probability variables for XGBoost
prob_train_XGBoost = np.full((np.shape(train)[0], outer_cv_folds), np.nan)
prob_test_XGBoost = np.full((np.shape(train)[0], outer_cv_folds), np.nan)
aucs_train_XGBoost = np.full(outer_cv_folds, np.nan)
aucs_test_XGBoost = np.full(outer_cv_folds, np.nan)

# Initialize probability variables for TabNet
prob_train_TabNet = np.full((np.shape(train)[0], outer_cv_folds), np.nan)
prob_test_TabNet = np.full((np.shape(train)[0], outer_cv_folds), np.nan)
aucs_train_TabNet = np.full(outer_cv_folds, np.nan)
aucs_test_TabNet = np.full(outer_cv_folds, np.nan)

# Use the function LabelEncoder for the column diagnosis
train['diagnosis'] = LabelEncoder().fit_transform(train['diagnosis'])
#Separate variables (x) from target (y) for XGBoost
x =  train.drop(['diagnosis'],axis='columns')
y = train.diagnosis

#Outer cross validation
cv_outer = StratifiedKFold(n_splits=outer_cv_folds)
ncv_idx = -1

for train_idx, test_idx in cv_outer.split(x, y):
    ncv_idx += 1
    train_data, test_data = x.iloc[train_idx], x.iloc[test_idx]
    train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]
    XGBoost_model = XGBClassifier(objective = 'binary:logistic', # Add objective and metric to model initialization
                                  eval_metric = 'auc')
    # Find best XGBoost parameters
    cv_inner = StratifiedKFold(n_splits=3)   # Training data being split 3 times 
    Parmt_XGBoost = {'n_estimators':[50, 100],
                     'max_depth':[3, 5],
                     'learning_rate':[0.01, 0.1, 0.3],
                     'colsample_bytree':[0.5, 1],
                     'gamma':[0],
                     }
    
    Parmt_model_XGBoost = GridSearchCV(estimator=XGBoost_model,
                                       param_grid=Parmt_XGBoost,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       cv=cv_inner).fit(train_data,train_target)
    best_parameters_XGBoost = Parmt_model_XGBoost.best_params_
    # Set best parameters to XGBoost and Tabnet model
    XGBoost_model.set_params(**best_parameters_XGBoost)
    # Train optimized XGBoost model on train data
    XGBoost_model.fit(train_data,train_target)
    # Train data results
    prob_train_XGBoost[train_idx, ncv_idx] = XGBoost_model.predict_proba(train_data)[:,1]
    aucs_train_XGBoost[ncv_idx] = metrics.roc_auc_score(train_target, prob_train_XGBoost[train_idx, ncv_idx])
    # Test data results
    prob_test_XGBoost[test_idx, ncv_idx] = XGBoost_model.predict_proba(test_data)[:,1]
    aucs_test_XGBoost[ncv_idx] = metrics.roc_auc_score(test_target, prob_test_XGBoost[test_idx, ncv_idx])
    
    # Find best TabNet parameters
    TabNet_model = TabNetClassifier(cat_idxs=[])
    
    TabNet_model.fit(X_train=train_data.to_numpy(),  
            y_train=train_target.to_numpy(), 
            eval_set=[(train_data.to_numpy(), train_target.to_numpy())], 
            eval_name=['train'], 
            eval_metric=['auc'],
            max_epochs=2000, 
            patience=2000,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False)

    # Train data results
    prob_train_TabNet[train_idx, ncv_idx] = TabNet_model.predict_proba(train_data.to_numpy())[:,1]
    aucs_train_TabNet[ncv_idx] = metrics.roc_auc_score(train_target, prob_train_TabNet[train_idx, ncv_idx])
    # Test data results
    prob_test_TabNet[test_idx, ncv_idx] = TabNet_model.predict_proba(test_data.to_numpy())[:,1]
    aucs_test_TabNet[ncv_idx] = metrics.roc_auc_score(test_target, prob_test_TabNet[test_idx, ncv_idx])

# Final test predictions in one column
prob_test_final_XGB = np.nanmean(prob_test_XGBoost, axis=1)
prob_test_final_TabNet = np.nanmean(prob_test_TabNet, axis=1)

# AUC based on test predictions
auc_test_prob_XGB = metrics.roc_auc_score(y, prob_test_final_XGB)
auc_test_prob_TabNet = metrics.roc_auc_score(y, prob_test_final_TabNet)

#Plot AUC
xgb_fpr, xgb_tpr, threshold = metrics.roc_curve(y, prob_test_final_XGB)
TabNet_fpr, TabNet_tpr, threshold = metrics.roc_curve(y, prob_test_final_TabNet)

auc_xgb = metrics.auc(xgb_fpr, xgb_tpr)
auc_TabNet_test = metrics.auc(TabNet_fpr, TabNet_tpr)

plt.figure(figsize=(5,5), dpi=100)
plt.plot(xgb_fpr, xgb_tpr, linestyle='-', label='xgb(auc=%0.3f)' % auc_xgb)
plt.plot(TabNet_fpr, TabNet_tpr, linestyle='dotted', label='TabNet(auc=%0.3f)' % auc_TabNet_test)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()