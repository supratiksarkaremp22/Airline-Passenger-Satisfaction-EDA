#!/usr/bin/env python
# coding: utf-8

# In[1]:


# we import the required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[2]:


# set the figure size for visualizations
sns.set(rc={'figure.figsize':(10,8)})


# In[3]:


# we import the dataset
df = pd.read_csv("airline_passenger_satisfaction.csv", index_col="Unnamed: 0")
df.head()


# In[4]:


# we check the shape
df.shape


# In[5]:


df.dtypes # we check data types


# In[6]:


df.isna().sum() # check nulls


# In[7]:


from scipy import stats
stats.pearsonr(df.dropna()["departure_delay_in_minutes"], df.dropna()["arrival_delay_in_minutes"])


# In[8]:


df['arrival_delay_in_minutes'].fillna(df['departure_delay_in_minutes'], inplace=True) # filled the missing values


# In[9]:


# pie chart of gender
plt.pie(x=df['Gender'].value_counts(), startangle=90, autopct='%1.0f%%')
plt.legend(title="Gender", loc="upper right", labels=["Female", "Male"])
plt.show()


# In[10]:


plt.figure(figsize=(10,2))
sns.countplot(y=df["customer_type"], palette="tab10");


# In[11]:


df["customer_type"].value_counts(normalize=True)*100


# In[12]:


skew = round(df["age"].skew(), 2)
kurt = round(df["age"].kurtosis(), 2)
print("Skewness:", skew, "\nKurtosis:", kurt)
sns.histplot(df["age"],bins=40);


# In[13]:


sns.displot(data=df, x="age", kind='kde', hue='satisfaction');


# In[14]:


plt.figure(figsize=(10,2))
sns.countplot(y=df["type_of_travel"], order=df["type_of_travel"].value_counts().index, palette="tab10");


# In[15]:


df["type_of_travel"].value_counts(normalize=True)*100


# In[16]:


plt.figure(figsize=(12,2))
sns.countplot(y=df["customer_class"], order=df["customer_class"].value_counts().index, palette="tab10");


# In[17]:


df["customer_class"].value_counts(normalize=True)*100


# In[18]:


skew = round(df["flight_distance"].skew(), 2)
kurt = round(df["flight_distance"].kurtosis(), 2)
print("Skewness:", skew, "\nKurtosis:", kurt)
sns.histplot(df["flight_distance"],bins=60);


# In[19]:


sns.boxenplot(df["flight_distance"])


# In[20]:


sns.displot(data=df, x="flight_distance", kind='kde', hue='satisfaction');


# In[21]:


df["satisfaction"].value_counts(normalize=True)*100


# In[22]:


# we plot the correlation matrix
plt.figure(figsize=(30,20))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues")
plt.show()


# In[23]:


# normalizing our features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(["satisfaction"], axis=1))
y = df["satisfaction"]


# In[24]:


# training and testing splits
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# In[25]:


# logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
model_training_score = round(100*model.score(X_train, y_train),2)
model_test_score = round(100*model.score(X_test, y_test),2)
print("Train Accuracy:", model_training_score, "%\nTest Accuracy:", model_test_score, "%")


# In[ ]:




