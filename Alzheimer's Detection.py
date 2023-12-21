#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn 
from sklearn import metrics
from sklearn import pipeline
from sklearn import svm
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[2]:


import seaborn as s


# In[3]:


df1=pd.read_csv("oasis_longitudinal.csv")


# In[4]:


df1.head()


# In[5]:


df1.tail()


# In[6]:


df1.shape


# In[7]:


df1.size


# In[8]:


df1.count()


# In[9]:


df1.value_counts()


# In[10]:


df1.isnull().sum()


# In[11]:


df1.columns


# In[12]:


df1['Subject ID']


# In[13]:


df1['MRI ID']


# In[14]:


df1['Group']


# In[15]:


df1['Visit']


# In[16]:


df1['MR Delay']


# In[17]:


df1['M/F']


# In[18]:


df1['Hand']


# In[19]:


df1['Age']


# In[20]:


df1['EDUC']


# In[21]:


df1['SES']


# In[22]:


df1['CDR']


# In[23]:


df1['eTIV']


# In[24]:


df1 = df1.drop(['Subject ID', 'MRI ID', 'Hand', 'Visit', 'MR Delay'], axis=1)


# In[25]:


df1.shape


# In[26]:


df1['Group'].value_counts()


# In[27]:


df1['Group'].count()


# In[28]:


df2=df1['Group'].replace(['Converted'],['Demented'])


# In[29]:


df2.count()


# In[30]:


df2.isnull().sum()


# In[31]:


df1['SES'] = df1['SES'].fillna(value= df1['SES'].mode().iloc[0])
df1['MMSE'] = df1['MMSE'].fillna(value =df1['MMSE'].median())


# In[32]:


X = df1[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]


# In[33]:


Y = df1['Group'].values


# In[34]:


Y


# In[35]:


ns = ['Age', 'EDUC',  'MMSE', 'eTIV', 'nWBV', 'ASF']
cs = ['M/F', 'SES']


# In[36]:


n=MinMaxScaler()
c = OneHotEncoder(drop='first')


# In[37]:


p = ColumnTransformer(transformers=[('num', n, ns),
                                    ('cat', c, cs)])


# In[38]:


s.countplot(data=df1, x='Group')
plt.title("Distribution of Classes")
plt.xlabel("Group")
plt.ylabel("Count")
plt.show()


# In[47]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)


# In[48]:


pl_rf = Pipeline(steps=[('preprocessor', p),('classifier', RandomForestClassifier(random_state=0))])
 
pl_rf.fit(X_train, Y_train)


# In[49]:


Y_pred_rf = pl_rf.predict(X_test)


# In[ ]:





# In[ ]:




