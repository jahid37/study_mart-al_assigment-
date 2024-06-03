#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# In[5]:


digits=load_digits()


# In[7]:


dir(digits)


# In[11]:


digits.data[0]


# In[14]:


plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])


# In[16]:


digits.target[0:5]


# In[17]:


from sklearn.model_selection import train_test_split 


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)


# In[26]:


len(x_train)


# In[28]:


len(x_test)


# In[31]:


from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()


# In[33]:


model.fit(x_train,y_train)


# In[36]:


model.score(x_test,y_test)


# In[ ]:


plt.matshow

