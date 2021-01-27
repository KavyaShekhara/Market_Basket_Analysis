#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[16]:


pwd


# In[19]:


data = pd.read_csv(r"C:\Users\kkavy\Desktop\Secret\Mandara\Market_Basket_Analysis\Breakfast_Basket.csv")


# In[21]:


data.head()


# In[23]:


data.shape


# In[27]:


data.isnull().sum()


# In[52]:


data['Item'].value_counts()


# In[55]:


data.dtypes


# In[57]:


data['Item'].nunique()


# In[59]:


data.head(1)


# In[61]:


data['date_time']=pd.to_datetime(data['date_time'])


# In[63]:


data.head(1)


# In[67]:


data['Transaction'].nunique()


# In[69]:


data.head()


# In[71]:


data.shape


# In[74]:


transactions_str = data.groupby(['Transaction','Item'])['Item'].count().reset_index(name = 'Count')
transactions_str


# In[77]:


my_basket = transactions_str.pivot_table(index = 'Transaction', columns = 'Item', values='Count', aggfunc = 'sum').fillna(0)
my_basket.tail()


# In[79]:


my_basket.head()


# In[81]:


def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1


# In[83]:


my_basket_sets = my_basket.applymap(encode)
my_basket_sets.tail()


# In[85]:


frequent_items = apriori(my_basket_sets, min_support = 0.01, use_colnames = True)
frequent_items


# In[86]:


association_rule = association_rules(frequent_items,metric = 'lift',min_threshold = 1)
association_rule

