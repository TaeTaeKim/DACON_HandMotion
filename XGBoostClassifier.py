#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
import pickle

import xgboost
from xgboost import XGBClassifier

import sklearn
from sklearn.model_selection import KFold,train_test_split,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')


# In[3]:


# Set directory & stick random seed for evaluation
WORKING_DIR = '/USER/DACON'

Trial_name = "mlp_scheduler_3rd"

RANDOM_SEED = 1001

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[4]:


# Load Raw data
train = pd.read_csv(os.path.join(WORKING_DIR,'train.csv')).drop('id',axis=1)
test = pd.read_csv(os.path.join(WORKING_DIR,'test.csv')).drop('id',axis=1)


# In[5]:


dataset = train.drop('target',axis=1)
target = train['target']


# In[6]:


# data normalization

#train data
norm_train = train.drop('target',axis=1)
norm_train = (train - train.mean()) / train.std()
norm_train['target'] = train['target']

# test data
norm_test = (test - test.mean()) / test.std()


# In[7]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# # Data Split

# In[8]:


X_train,X_valid,y_train,y_valid = train_test_split(dataset,target,
                                                  test_size=.2, random_state=RANDOM_SEED)


# # Xgboost 파라미터 조정

# In[10]:


model = XGBClassifier(random_state =RANDOM_SEED, eval_metric='mlogloss',
                      objective='multi:softmax',use_label_encoder=False,
                      tree_method='gpu_hist',predictor='gpu_predictor'
                     )

cv = RepeatedStratifiedKFold(random_state=RANDOM_SEED,n_splits=2)

param_grid = {
    'learning_rate':[.05,.01,.0001],
    'gamma':[0,0.1,0.2,.5],
    'max_depth': [3,5,7,10,20],
    'min_child_weight':[0,1,2],
    'subsample': [0.5,1],
    'colsample_bytree':[0.5,1],
    "n_estimators":[100,300,500,800,1000,1200,1400]
             }

grid_search = GridSearchCV(model, param_grid=param_grid,cv=cv,scoring='accuracy',verbose=3)
result = grid_search.fit(dataset,target)


# In[34]:


print(result.best_score_)


# In[35]:


optim_model = result.best_estimator_
optim_model.fit(dataset,target)


# In[36]:


with open('./weight/xgboost_model.pkl','wb') as file:
    pickle.dump(optim_model,file)


# In[38]:


y_pred = optim_model.predict(test)
print(y_pred.shape)


# In[40]:


submit = pd.read_csv('./sample_submission.csv')
submit['target']= y_pred
submit.to_csv(f'./result/xgboost_submit.csv',index=False)


# In[ ]:


print("Xgboost Task Finished")

