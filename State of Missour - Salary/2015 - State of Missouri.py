
# coding: utf-8

# ## Predicting Employee Pay at the individual level for Missouri state employees in 2015
# 
# The following dataset as been used for the following analysis:
# 
# https://data.mo.gov/Government-Administration/2015-State-Employee-Pay-As-Of-COB-November-30-2015/snq5-idu6

# In[82]:

# The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd

# Import the stats librayr from numpy
from scipy import stats

# These are the plotting modules adn libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Command so that plots appear in the iPython Notebook
get_ipython().magic(u'matplotlib inline')


# ###### Step 1: retrieving the data.

# In[10]:

#The exported file has been downloaded and renamed as 'Salary.csv'. Let's read it into a CSV file
salary_df = pd.read_csv('Salary.csv')


# In[3]:

#Let's take a sneak peek into the data
salary_df.head(20)


# Observations: 
# We have the following fields at hand:
# - Calendar Year	: Since all the data in the dataset is of the same year, this won't be of any help in predicting the data
# - Agency Name: This field could give us some useful insight
# - Position Title: This field could give us some useful insight
# - Employee Name: This field won't be of any use for predicting the pay
# - YTD Gross Pay: This field is our target 
# 
# So let's now drop the fields we won't be using in predicting the pay

# In[14]:

salary_cond_df = salary_df[[1,2,4]] #condensed dataframe
salary_cond_df.head()


# In[26]:

#let's see the unique values for Agency Name and Position Title
salary_cond_df[[0,1]].describe()


# As we can see there are 24 unique Agency Name. However there are 2601 unique Position Titles. That is a huge number.
# On a closer observation we see that many postions have multiple levels to them.For example "ACCOUNTANT" has positions like
# "ACCOUNTANT I", "ACCOUNTANT II" and so on.
# It would be worthwhile to seperate the Postion name and their level into seperate fields. The following cell does that 
# 
# Roman Suffixes like I, II etc.. are given a value 1,2,.. in the field 'Position Level'. Positions without the suffixes are given a value 0. 'Position Title New' contains the Position Title  without roman suffixes. 

# ##### Step 2: building features 

# In[53]:

position_lvl = pd.Series()
position_new = pd.Series()
lvl_dict = {"I " : 1, "II ": 2, "III " : 3, "VI ": 4, "V ": 5, "IV ": 6, "IIV ": 7}  #since the comparasion is with reverse
def get_position_lvl(sal_df):
    for idx,pos in sal_df.iterrows():
        new_pos = pos[1]
        temp = 0
        if pos[1][::-1][:2] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:2]]
            new_pos = new_pos[0:len(new_pos) - 2]
        elif pos[1][::-1][:3] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:3]]
            new_pos = new_pos[0:len(new_pos) - 3]
        elif pos[1][::-1][:4] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:4]]
            new_pos = new_pos[0:len(new_pos) - 4]        
        position_new.set_value(idx,new_pos )
        position_lvl.set_value(idx,temp )


# In[59]:

get_position_lvl(salary_cond_df)
salary_cond_df['Position Title New'] =  position_new
salary_cond_df['Position Level'] = position_lvl


# In[61]:

#Let's take a look at our new data
salary_cond_df.head(100)


# #### Hunting for outliers
# 
# In this step we will do the following two things:
# 1) Find and delete any salary that does not fall into the normal range (i.e. too large)
# 2) Any negitive salary

# In[67]:

#First let's sort the Gross Pay in decending order to see any outliers
sorted(salary_cond_df['YTD Gross Pay'], reverse = True)


# In[72]:

#As we see there's one salary that's far greater than other salaries. Let's find and deleted it
salary_cond_df[salary_cond_df['YTD Gross Pay']>400000]


# In[75]:

salary_cond_df.drop(31627, inplace=True)
sorted(salary_cond_df['YTD Gross Pay'], reverse = True)


# In[78]:

#Now let's delete all the salaries that are less than zero
salary_cond_df.drop(salary_cond_df[salary_cond_df['YTD Gross Pay'] <0].index , inplace=True)


# In[79]:

sorted(salary_cond_df['YTD Gross Pay'])


# #### Rounding off 
# Now let's convert all the Float values into Real numbers. This will make it easy for us to visualize the data

# In[98]:

salary_cond_df=salary_cond_df.round({'YTD Gross Pay': 0})


# In[99]:

salary_cond_df.head()


# At this point, it is very obvious that the number of features are too less. We only have two features : Agency Name, Position Title that can somehow predict the Gross Pay.

# ###### Step 3: Predicitng the salary
# 
# One simple way to predict that salary is to create two pivot tables:
# 
# 1. Of mean salary
# 
# 2. Of Standard deviation.
# 
# Using the information of these two tables, we can predict the Range of the salary of an employee given his or her Agency Name and  Position Title
# 

# In[102]:

#Mean
pd.pivot_table(salary_cond_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], 
               aggfunc=np.mean).dropna(how='all')


# In[103]:

#Standard deviation
pd.pivot_table(salary_cond_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], 
               aggfunc=np.std).dropna(how='all')


# Using the above two tables, we can with 68.2 confidence predict the range of salary of a person.
# For example, someone with Agency Name = 'CORRECTIONS' and Position Title ='ACADEMIC TEACHER III'
# would have a salary between (28113.786325 -7762.997396) to  (28113.786325 + 7762.997396) with 68% confidence.
# 
# This is ofcourse, a huge range stil.

# #### Visualizing data 
# Let's try to visualize the Gross Pay against the Position Level and see if we can find any linear pattern 

# In[121]:

sns.lmplot("Position Level","YTD Gross Pay",salary_cond_df)


# As we can see there isn't any obvious pattern between the Gross Pay and Position Levels. As such, it's very hard to predict salary based on Position Level . We can try to go higher order to see if we can find any relationship. We can also draw a vilon plot to get a better understanding .

# In[ ]:




# In[130]:

# Create figure with 2 subplots
fig, (axis1,axis2) = plt.subplots(1,2,sharey =True)

sns.regplot("Position Level","YTD Gross Pay",salary_cond_df, order =5, ax = axis1)
sns.violinplot(salary_cond_df['Position Level'],salary_cond_df['YTD Gross Pay'],order=[0,1,2,3,4,5,6],ax=axis2)


# As we can now see, this gives a slightly better understanding of the relationship between Poistion level and Gross Pay, however it's still not enough for us to predict the salary.

# #### Approaches to improve the model
# 
# Because the number of features are so less, regression ML algorithms won't be of much use. Infact, the only viable option is to improve the data quality and incorporate more features like : "Experience of Person" "Performance Rating" etc. 

# In[ ]:



