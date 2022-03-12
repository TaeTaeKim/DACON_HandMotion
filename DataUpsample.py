#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import random
import numpy as np
from tqdm import tqdm


# In[93]:


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[94]:


train.drop(columns='id',inplace=True)


# In[95]:


train.head()


# In[96]:


label_0 = train.loc[train['target']==0,:].to_numpy()
label_1 = train.loc[train['target']==1,:].to_numpy()
label_2 = train.loc[train['target']==2,:].to_numpy()
label_3 = train.loc[train['target']==3,:].to_numpy()


# In[97]:


label_list = [label_0,label_1,label_2,label_3]


# In[104]:


def hand_upsample(dataset,num_upsample):
    new_dataset=[]
    for _ in tqdm(range(num_upsample)):
        new_row = []
        for i in range(33):
            new_row.append(random.choice(dataset[:,i]))
        new_dataset.append(new_row)
    return pd.DataFrame(new_dataset,columns=train.columns)
        
    
        


# In[105]:


new_train = pd.DataFrame(columns=train.columns)
for label in label_list:
    new_set = hand_upsample(label,num_upsample=2000)
    new_train = pd.concat([new_train,new_set],axis=0,ignore_index=True)
new_train['target'] = new_train['target'].astype('int64')


# In[106]:


new_train.to_csv('Upsample_train.csv',index=False)


# In[ ]:




