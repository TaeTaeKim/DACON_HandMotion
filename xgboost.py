#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import random
from datetime import datetime

from xgboost import XGBClassifier

import sklearn
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import accuracy_score


# # XGBoost link
# (https://dacon.io/competitions/official/235876/codeshare/4664?page=1&dtype=recent)

# In[110]:


# Set directory & stick random seed for evaluation
WORKING_DIR = '/USER/DACON'

Trial_name = "mlp_scheduler_3rd"

RANDOM_SEED = 1001
Patient = 50
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[92]:


# Load Raw data
train = pd.read_csv(os.path.join(WORKING_DIR,'train.csv')).drop('id',axis=1)
test = pd.read_csv(os.path.join(WORKING_DIR,'test.csv')).drop('id',axis=1)


# In[93]:


dataset = train.drop('target',axis=1)
target = train['target']
testset = torch.FloatTensor(test.to_numpy()).cuda()


# In[94]:


# data normalization

#train data
norm_train = train.drop('target',axis=1)
norm_train = (train - train.mean()) / train.std()
norm_train['target'] = train['target']

# test data
norm_test = (test - test.mean()) / test.std()


# In[95]:


# Train parameter

EPOCH = 500
BATCH_SIZE = 4


# In[96]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# # Data Split

# In[97]:


X_train,X_valid,y_train,y_valid = train_test_split(dataset,target,
                                                  test_size=.2, random_state=RANDOM_SEED)

