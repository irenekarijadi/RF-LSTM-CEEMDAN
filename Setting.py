#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Parameters Settings

max_features=8 # the feature number of each node is set to 8, as it is suggested to be one-third of the feature's number (Lahouar and Slama,2017)
lr=0.001 #learning rate LSTM as recommended in (Kingma and Ba, 2014) 
optimizer='Adam' #Adam is used as the optimizer for LSTM as it is computationally efficient (Dubey, Ashutosh Kumar, et al,2021)
neuron=64 
epoch=100 #(Jorges et al, 2021)
batch_size=64 #(Kandel et al,2020)
n_hours=24 #in this study, to predict one hour ahead predicton, we use input from the previous 24 hour energy consumption 
data_partition=0.8 #in this study, we divided the data into 80% of data as training data and 20% of the data as testing data


# In[ ]:




