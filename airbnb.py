
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # * A walkthrough data analyzing and cleaning with Pandas.
# # * This is Airbnb New Users Booking data a part of Kaglle comptetions data it's avilable on kaggle website.
# # * I will do another part follow, that will cover visualzation with seaborn and machine learning model in the future. 
# 

# In[2]:


# Get work directory in the system
import os
os.getcwd()


# In[3]:


# read raw data into pandas
air = pd.read_csv('train_users_2.csv')


# In[4]:


air.shape


# In[5]:


# a quick look into first 10 rows
air.head(10)


# In[6]:


air.dtypes


# # As the shape of data shown above that 16 columns of more than tow hundred thousands of data available.And at the first glance of data can tell there are a lot of missing data. 
# let's review the data :
# \> In this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics.
# 
# \> All the users in this dataset are from the USA.
# 
# \> There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. 
# 
# \> Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.
# 
# 
# \> The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.) 
# 

# # Also before conuntinue any further there is some qoustion in mind:
# 1. What type of formate for date, numerical values, are provided 
# 2. How about the missing values ?
# 3. How do the diffrerent features relate to each other ?

# In[7]:


air.info()


# In[8]:


type(air)


# In[9]:


air.columns


# In[10]:


# Check up the missing values
air.isnull().sum()


# In[11]:


# All focusing will be on the train data set therefore let's dive in..
air.head(20)


# # Looking at the data above it provide us with key information :
#  1. What type of formate for date, numerical values, are provided 
#  2. How about the missing values ?
#  3. How do the diffrerent features relate to each other ?

# In[12]:


air.country_destination.describe()


# In[13]:


# There are 12 possible outcomes of the destination country. 
sorted(air.country_destination.unique())


# In[14]:


air[air.country_destination.isin(['NDF'])]


# In[15]:


# The chart belew users by destination. 
a = air.country_destination.value_counts()
b = air.country_destination.value_counts(normalize=True)


# In[16]:


# users by destination in both numbers and percentage.
pd.concat([a, b], axis=1)


# # The most important colmun among the dataset is the Country Destination column thus is the one model will try to predict, by looking at the codes above the number of records of each destination can help provide some insight and how our model going to be constructed.
# 

# In[17]:


air['date_account_created'].value_counts


# In[18]:



pd.crosstab(air.date_account_created, air.country_destination)


# In[19]:


pd.crosstab(air.country_destination, air.date_account_created)


# In[20]:


air.head(1)


# In[21]:


air.iloc[:, [1, 15]].sort_values('date_account_created', ascending=True, kind='quicksort')


# In[22]:


air.date_account_created.value_counts()


# In[23]:


air.loc[:, 'date_account_created']


# # In my opinion one of the most important part of analyzing is to spend some time digging into the data, thus all of sudden an intersing insight could  appear. 

# In[24]:


air.head()


# In[25]:


air.set_index('country_destination', inplace=True)


# In[26]:


air.head()


# In[27]:


# this move here by shifting index will simplefy to create new column call year which extracted out of date account created.
air.index.name = 'country_destination'
air.reset_index(inplace=True)
air.head()


# In[28]:


# our new year column will allow us to make usfull comparison. 
air.set_index('country_destination', inplace=True)
air.head()


# In[29]:


air.date_account_created.head()


# In[30]:


air.dtypes


# In[31]:


air['date_account_created'] = pd.to_datetime(air.date_account_created)


# In[32]:


air['timestamp_first_active'] = pd.to_datetime(air.timestamp_first_active)


# In[33]:


air.head()


# In[34]:


air.date_account_created.dt.year.value_counts()


# In[35]:


air.date_account_created.dt.year.sort_index()


# In[36]:


air.index.name = 'country_destination'
air.reset_index(inplace=True)
air.head()


# In[37]:


air['year']= air.date_account_created.dt.year


# In[38]:


air.head()


# In[39]:


matplotlib inline


# In[40]:


air.year.value_counts(normalize=True)


# In[41]:


air.year.value_counts().sort_values().plot(kind='bar', legend=True)


# In[42]:


air.country_destination.value_counts().sort_index().plot(kind='barh', legend=True)


# # the breackdown of the data stands out into obviouse fact:
# # 90% of users fall into tow catgories, they either yet make a booking (NDF) or they made thier first booking in the US and also the precentaage of making booking increase each year and reached over 60% in 2014.

# In[43]:


# the chart here confirm that most of the user majority fall into tow catigorize 
air.groupby('year').country_destination.value_counts().unstack().plot(legend=True)


# In[44]:


# This chart provide an estimation of each country destination that book from by years .
air.groupby('year').country_destination.value_counts().unstack()


# # what happened that the splits above will be benifital for modiling later on,  and it also  means couple of things:
# # first considering that our final prediction will be made against all the new users with first activites after 2014.
# # also as we saw in the country destination chart the vast majority of the data are in tow catgories and for that reason need to make ensure the training data has enugh information to ensure our model prediction.
# 
# # (In the test set, you will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.) 

# In[45]:


air.head()


# In[46]:


air.iloc[:, [0, 16]].sort_values('year').sort_index().describe()


# In[47]:


air[air.date_account_created > '2014']


# In[48]:


air.groupby('year').date_account_created.describe()


# In[49]:


air.groupby('date_account_created').year.plot()


# # Now looking at the chart it's provides woundefull proofs of the tremondus growth of Airbnb monthly of new accounts opened and the number of account has been created in the year of 2014. in the year to June 2014 the number of account created was 125,884- 132% increase from the year before.
# # Besides showing how quicky Airnbn has grown, the data also another important facts that the vast malority of the data comes from the last tow years. this maatter because  Airbnb, if we want to use the data in sessions.csv we would be limited to data from January 2014 onwards. Again looking at the numbers, this means that even though the sessions.csv data only covers 11% of the time period (6 out of 54 months), it still covers over 30% of the training data – or 76,466 users.

# In[50]:


air.date_account_created.value_counts().plot(kind='line', legend=True)


# # another wounderfull evidence of the growth increasing of Airnub averging over 10% percent growth in new account created per month.in the year of 2014 the number of new account 132% increase form the year before.
# # this data also provide important insight, the majority of the training data provided come from the latest 2 year....

# In[51]:


air.age.describe()


# In[52]:


air.columns


# In[53]:


air.age.isnull().sum()


# In[54]:


sorted(air.country_destination.unique())


# In[55]:


air.groupby('age').year.describe().plot()


# In[56]:


air.age.isnull().sum()


# In[57]:


air.age.describe().mean


# In[58]:


air.isnull().sum()


# # As we mentions before about the missing values need to be corrected we got three columns of missing information and out of the three it appears that only one is important for the analyzing.  

# In[59]:


air.age.fillna('49').head()


# In[60]:


air["age"].fillna(air.groupby("year")["age"].transform("median"), inplace=True)


# In[61]:


air.age.head()


# In[62]:


air.age.describe()


# # there are a lot of opthions of considration when dealing with missing values whether its nominal / catigorical or numrical / ordinal values and the key to sucess it deapends on the data itself. in this airbnb data choosing the mean to fill in with the missing values.

# In[63]:


air.age.describe()


# In[64]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


air.columns


# In[ ]:


air.groupby('first_device_type').year.value_counts(normalize=True, ascending=True).unstack()


# # looking into the decvice types shows how diffrient devices has been used across and which one has took off the battle. mac Desktop wining across the year follwed by windows user and there is signficantly increasiing.
# # as with the other columns we have reviewed above, this change over time reinforces the presumption that recent data is likely to be the most useful for building our model.

# In[ ]:


air.groupby('country_destination').year.sum()


# In[ ]:


air.signup_method.unique()


# In[ ]:


air.groupby('year').country_destination.value_counts().unstack()


# In[ ]:


air.head()


# # ok we already discovered an intersing meangfull insight through out the data thus, when peaple talking about cleaning data there are a few specfice things:
# 1- fixing datetime foramte.
# 2- filling in missing vlaues
# 3- correcting misinterpreted values
# 
