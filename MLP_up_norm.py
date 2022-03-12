#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
import numpy as np
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sklearn
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import accuracy_score


# In[41]:


# Set directory & stick random seed for evaluation
WORKING_DIR = '/USER/DACON'

Trial_name = "Upsample_mlp_norm"
mode = "norm" # 정규화된 데이터로 할 것인지 일반 데이터로 할것인지
debug = False

RANDOM_SEED = 1001
Patient = 50
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[42]:


# Load Raw data
train_df = pd.read_csv(os.path.join(WORKING_DIR,'train.csv')).drop('id',axis=1)
if 'Upsample' in Trial_name:
    train_df = pd.concat([train_df,pd.read_csv('Upsample_train.csv')],axis=0,ignore_index=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test = pd.read_csv(os.path.join(WORKING_DIR,'test.csv')).drop('id',axis=1)


# In[43]:


# train dataset
if mode=='basic':    
    target = train_df['target']
    train = train_df.drop('target',axis=1)
    testset = torch.FloatTensor(test.to_numpy()).cuda()
else:    
    target = train_df['target']
    train = train_df.drop('target',axis=1)
    train = (train - train.mean()) / train.std()
    
    
    testset = (test - test.mean()) / test.std()
    testset = torch.FloatTensor(testset.to_numpy()).cuda()


# In[45]:


# Train parameter

EPOCH = 500
BATCH_SIZE = 4


# In[46]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# # Data Split

# In[47]:


X_train,X_valid,y_train,y_valid = train_test_split(train,target,test_size=.2,
                                                   random_state=RANDOM_SEED,
                                                   stratify=target)


# In[48]:


len(X_train), len(X_valid)


# # DataSet

# In[49]:


class hand_dataset(Dataset):
    def __init__(self,X_data,y_data):
        self.X_data = X_data
        self.y_data = y_data
    
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]
    
    def __len__(self):
        return len(self.X_data)


# In[50]:


trainset = hand_dataset(torch.FloatTensor(X_train.to_numpy()),torch.LongTensor(y_train.to_numpy()))
validset = hand_dataset(torch.FloatTensor(X_valid.to_numpy()),torch.LongTensor(y_valid.to_numpy()))


# In[51]:


train_loader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(validset, batch_size = BATCH_SIZE, shuffle=False)


# In[53]:


if debug:
    for data,label in train_loader:
        print(data)
        print(label)
        break


# # Model Structure & Load

# In[54]:


class hand_model(nn.Module):
    def __init__(self,num_feature, num_classes):
        super(hand_model,self).__init__()
        
        self.Layer = nn.Sequential(
            nn.Linear(num_feature, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),


            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
    
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, num_classes)
        )
        
    def forward(self,x):
        x = self.Layer(x)
        return x


# In[55]:


model = hand_model(num_feature=32, num_classes=4).cuda()


# In[56]:


print(model)


# # Train Function

# In[57]:


def train(train_loader, valid_loader,verbose):
    best_loss = 100
    cnt = 0
    for epoch in range(EPOCH):
        start_time = timer()
        train_loss= 0 
        train_acc = 0
        val_loss = 0
        val_acc = 0
        model.train()
        for i,(X_batch,y_batch) in enumerate(train_loader):

            X_batch,y_batch = X_batch.cuda(), y_batch.cuda()
            
            output = model(X_batch)
            loss = criterion(output,y_batch)
            
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            
            train_loss+= loss.item()
            y_pred = torch.max(output,1).indices
            
            train_acc += accuracy_score(y_batch.cpu(),y_pred.cpu())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid, y_valid = X_valid.cuda(), y_valid.cuda()
                
                output = model(X_valid)
                loss = criterion(output,y_valid)
                val_loss += loss.item()
                y_pred = torch.max(output,1).indices
                val_acc += accuracy_score(y_valid.cpu(),y_pred.cpu())
        train_loss = train_loss/ len(train_loader)
        train_acc = train_acc / len(train_loader)
        val_loss = val_loss / len(valid_loader)
        val_acc  = val_acc / len(valid_loader)
        if verbose == 1:
            print(
                f'Epoch [ {epoch+1}/{EPOCH} ] Train_loss : [{train_loss:.6f}] Train_acc : [{train_acc}]\n Valid_loss : [{val_loss:.6f}] Valid_acc : [{val_acc}]',end=" "
            )
            if val_loss<best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(),os.path.join(WORKING_DIR,'weight',f'{Trial_name}.pth'))
                print(f"model saved best_loss [{best_loss}]")
                cnt =0
            else:
                cnt +=1
                if cnt== Patient:
                    print("Train Early stopped")
                    break
                else:
                    print(f'Early Stop : [{cnt}/{Patient}] best_loss : [{best_loss}]')
        elif verbose ==0:
            if val_loss<best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(),os.path.join(WORKING_DIR,'weight',f'{Trial_name}.pth'))
                print(f"Epoch [ {epoch+1}/{EPOCH} ] Val_loss:[{val_loss}] Val_acc : [{val_acc}] model saved -> best_loss = [{best_loss}]")
                cnt =0
            else:
                cnt +=1
                if cnt== Patient:
                    print("Train Early stopped")
                    break
                else:
                    if cnt%10==0:
                        print(f'Early Stop : [{cnt}/{Patient}]')
                


# In[58]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0000001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,epochs=EPOCH, steps_per_epoch=len(train_loader))


# In[59]:


train(train_loader,valid_loader,verbose=1)


# In[118]:


model.load_state_dict(torch.load(f'./weight/{Trial_name}.pth'))


# In[119]:


with torch.no_grad():
    model.eval()
    result = model(testset).cpu()
    pred =torch.max(result,1).indices.cpu()


# In[120]:


submit = pd.read_csv('./sample_submission.csv')


# In[121]:


submit['target']= pred


# In[122]:


submit.to_csv(f'./result/{Trial_name}_submit.csv',index=False)


# # 데이콘 제출 링크
# * [링크](https://dacon.io/competitions/official/235876/leaderboard)best_loss
