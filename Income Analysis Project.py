#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\adult_data.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data.corr()


# In[10]:


import sklearn


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


le=LabelEncoder()


# In[13]:


for i in range(0,data.shape[1]):
    if data.dtypes[i]=="object":
        data[data.columns[i]]=le.fit_transform(data[data.columns[i]])


# In[14]:


data.head()


# In[15]:


x=data.drop([" salary"],axis=1)


# In[16]:


y=data[" salary"]


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


sc=StandardScaler()


# In[19]:


x=sc.fit_transform(x)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[22]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


dc=DecisionTreeClassifier(criterion='gini',max_depth=100)


# In[25]:


dc.fit(x_train,y_train)


# In[26]:


pred=dc.predict(x_test)


# In[27]:


pred[0:5]


# In[28]:


y_test[0:5]


# In[29]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[30]:


print(confusion_matrix(pred,y_test))


# In[31]:


print(classification_report(pred,y_test))


# In[32]:


accuracy_score(pred,y_test)


# In[33]:


from sklearn.naive_bayes import BernoulliNB


# In[34]:


bl=BernoulliNB()


# In[35]:


bl.fit(x_train,y_train)


# In[36]:


pred2=bl.predict(x_test)


# In[37]:


print(classification_report(pred2,y_test))


# In[38]:


print(accuracy_score(pred2,y_test))


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


rf=RandomForestClassifier(n_estimators=100,criterion='entropy',)


# In[41]:


rf.fit(x_train,y_train)


# In[42]:


pred3=rf.predict(x_test)


# In[43]:


pred3[0:5]


# In[44]:


y_test[0:5]


# In[45]:


print(confusion_matrix(pred3,y_test))
print(classification_report(pred3,y_test))
print(accuracy_score(pred3,y_test))


# In[46]:


from sklearn.ensemble import AdaBoostClassifier


# In[47]:


ad=AdaBoostClassifier(n_estimators=200)


# In[48]:


ad.fit(x_train,y_train)


# In[49]:


pred4=ad.predict(x_test)


# In[50]:


print(confusion_matrix(pred4,y_test))
print(classification_report(pred4,y_test))
print(accuracy_score(pred4,y_test))


# In[51]:


from sklearn.svm import SVC


# In[52]:


sc=SVC()


# In[53]:


sc.fit(x_train,y_train)


# In[54]:


pred5=sc.predict(x_test)


# In[55]:


print(confusion_matrix(pred5,y_test))
print(classification_report(pred5,y_test))
print(accuracy_score(pred5,y_test))


# In[56]:


data1=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\adult_test.csv")


# In[57]:


data1.head()


# In[58]:


data1.isnull().sum()


# In[59]:


data1.tail()


# In[ ]:





# In[60]:


data1.head()


# In[64]:


print(data1.columns)


# In[66]:


data1[' workclass']=data1[' workclass'].str.replace("?"," ")


# In[69]:


data1.head()


# In[70]:


data1[' occupation']=data1[' occupation'].str.replace("?"," ")


# In[72]:


data1.head()


# In[73]:


data1.isnull().sum()


# In[74]:


data1.corr()


# In[75]:


from sklearn.preprocessing import LabelEncoder


# In[76]:


le=LabelEncoder()


# In[77]:


for i in range(0,data1.shape[1]):
    if data1.dtypes[i]=="object":
        data1[data1.columns[i]]=le.fit_transform(data1[data1.columns[i]])


# In[78]:


data1.head()


# In[79]:


x=data1.drop([" salary"],axis=1)


# In[80]:


y=data1[" salary"]


# In[81]:


from sklearn.preprocessing import StandardScaler


# In[82]:


sc=StandardScaler()


# In[83]:


x=sc.fit_transform(x)


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[86]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[87]:


from sklearn.tree import DecisionTreeClassifier


# In[88]:


dc=DecisionTreeClassifier(criterion='gini',max_depth=100)


# In[89]:


dc.fit(x_train,y_train)


# In[90]:


pred=dc.predict(x_test)


# In[91]:


pred[0:5]


# In[92]:


y_test[0:5]


# In[93]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[94]:


print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
print(accuracy_score(pred,y_test))


# In[95]:


from sklearn.ensemble import RandomForestClassifier


# In[96]:


rf=RandomForestClassifier(n_estimators=100,criterion='entropy',)


# In[97]:


rf.fit(x_train,y_train)


# In[98]:


pred1=rf.predict(x_test)


# In[99]:


print(confusion_matrix(pred1,y_test))
print(classification_report(pred1,y_test))
print(accuracy_score(pred1,y_test))


# In[100]:


from sklearn.ensemble import AdaBoostClassifier


# In[101]:


ad=AdaBoostClassifier(n_estimators=200)


# In[102]:


ad.fit(x_train,y_train)


# In[103]:


pred2=ad.predict(x_test)


# In[104]:


print(confusion_matrix(pred2,y_test))
print(classification_report(pred2,y_test))
print(accuracy_score(pred2,y_test))


# In[105]:


from sklearn.svm import SVC


# In[106]:


sc=SVC()


# In[107]:


sc.fit(x_train,y_train)


# In[108]:


pred3=sc.predict(x_test)


# In[109]:


print(confusion_matrix(pred2,y_test))
print(classification_report(pred2,y_test))
print(accuracy_score(pred3,y_test))


# In[ ]:




