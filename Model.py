#!/usr/bin/env python
# coding: utf-8

# # Fake news Detection

# ![I-Newspaper2.jpg](attachment:I-Newspaper2.jpg)

# ### Importing required library
# Here I am going to importing some of the required library, if extra library is required to install It will be install later on.

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# ### Inserting fake and real dataset

# In[3]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[4]:


df_fake.head(5)


# In[5]:


df_true.head(5)


# Inserting a column called "class" for fake and real news dataset to categories fake and true news. 

# In[6]:


df_fake["class"] = 0
df_true["class"] = 1


# Removing last 10 rows from both the dataset, for manual testing  

# In[7]:


df_fake.shape, df_true.shape


# In[8]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[9]:


df_fake.shape, df_true.shape


# Merging the manual testing dataframe in single dataset and save it in a csv file

# In[10]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[11]:


df_fake_manual_testing.head(10)


# In[12]:


df_true_manual_testing.head(10)


# In[13]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# Merging the main fake and true dataframe

# In[14]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[15]:


df_marge.columns


# #### "title",  "subject" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.

# In[16]:


df = df_marge.drop(["title", "subject","date"], axis = 1)


# In[17]:


df.isnull().sum()


# #### Randomly shuffling the dataframe 

# In[18]:


df = df.sample(frac = 1)


# In[19]:


df.head()


# In[20]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[21]:


df.columns


# In[22]:


df.head()


# #### Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.

# In[23]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[24]:


df["text"] = df["text"].apply(wordopt)


# #### Defining dependent and independent variable as x and y

# In[25]:


x = df["text"]
y = df["class"]


# #### Splitting the dataset into training set and testing set. 

# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# #### Convert text to vectors

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[28]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# ### 4. Random Forest Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[31]:


pred_rfc = RFC.predict(xv_test)


# In[32]:


RFC.score(xv_test, y_test)


# In[33]:


# print(classification_report(y_test, pred_rfc))


# In[37]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)

    return output_lable(pred_RFC[0])

# In[39]:

news = str(input())
manual_testing(news) 

