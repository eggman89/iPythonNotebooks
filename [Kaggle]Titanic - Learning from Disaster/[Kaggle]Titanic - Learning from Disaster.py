
# coding: utf-8

# ### Titanic 
# #### Learning from Disaster
# Check out the Kaggle Titanic Dataset at the following link:
# 
# https://www.kaggle.com/c/titanic/data
# 
# 

# In[1]:

# Importing essential Python libraries

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

titanic_df = pd.read_csv('train.csv')


# In[3]:

#Let's take a preview of the data
titanic_df.head()


# In[4]:

#we're missing a lot of cabin info
titanic_df.info()


# ###### First some basic questions:
# 
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 
# 2.) What deck were the passengers on and how does that relate to their class?
# 
# 3.) Where did the passengers come from?
# 
# 4.) Who was alone and who was with family?
# 
# ######  Then we'll dig deeper, with a broader question:
# 
# 5.) What factors helped someone survive the sinking?

# In[7]:

#1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)

# Let's first check gender
sns.factorplot('Sex',data=titanic_df,kind='count')


# In[13]:

# Now let's seperate the genders by classes, remember we can use the 'hue' arguement here!

sns.factorplot('Sex',data=titanic_df,hue='Pclass', kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Sex', kind='count')


# In[15]:

# We'll treat anyone as under 16 as a child
#  a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex


# In[17]:

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[19]:

titanic_df.head(10)


# In[22]:

# Let's try the factorplot again!
sns.factorplot('Pclass',data=titanic_df,hue='person', kind='count')


# In[23]:

#age : histogram using pandas
titanic_df['Age'].hist(bins=70)


# In[24]:

#Mean age of passengers
titanic_df['Age'].mean()


# In[25]:

titanic_df['person'].value_counts()


# In[27]:

fig = sns.FacetGrid(titanic_df, hue="Sex",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[29]:

fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[30]:

# Let's do the same for class by changing the hue argument:
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# We've gotten a pretty good picture of who the passengers were based on Sex, Age, and Class. So let's move on to our 2nd question: What deck were the passengers on and how does that relate to their class?

# In[32]:

#Dropping null values
deck = titanic_df['Cabin'].dropna()


# In[36]:

deck.head()


# In[37]:

#Notice we only need the first letter of the deck to classify its level (e.g. A,B,C,D,E,F,G)

levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    


# In[48]:

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d', kind= 'count', order=['A','B','C','D','E','F'])


# In[51]:

#Note here that the Embarked column has C,Q,and S values. 
#Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',x_order=['C','Q','S'], kind = 'count')


# An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.
# 
# Now let's take a look at the 4th question:
# 
# 4.) Who was alone and who was with family?

# In[54]:

# Let's start by adding a new column to define alone

# We'll add the parent/child column with the sibsp column
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp

# Look for >0 or ==0 to set alone status
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

titanic_df['Alone'].head()


# In[62]:

sns.factorplot('Alone',hue= 'Pclass',data=titanic_df,palette='Blues', kind = 'count')
sns.factorplot('Pclass',hue= 'Alone',data=titanic_df,order= [1,2,3], palette='Blues', kind = 'count')


# Now that we've throughly analyzed the data let's go ahead and take a look at the most interesting (and open-ended) question: What factors helped someone survive the sinking?
# 

# In[64]:

# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.factorplot('Survivor',data=titanic_df,palette='Set1', kind = 'count')


# So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate, since the movie Titanic popularized the notion that the 3rd class passengers did not do as well as their 1st and 2nd class counterparts.
# 

# In[67]:

# Let's use a factor plot again, but now considering class
sns.factorplot('Pclass','Survived',data=titanic_df, order = [1,2,3])


# In[69]:

# Let's use a factor plot again, but now considering class and gender
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,order = [1,2,3])


# But what about age? Did being younger or older have an effect on survival rate?

# In[73]:

# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass',palette='winter')


# In[74]:

# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[76]:

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)



# 1.) Did the deck have an effect on the passengers survival rate? 

# In[85]:

levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

cabin_df = cabin_df[cabin_df.Cabin != 'T']
titanic_df['Level'] = Series(levels,index=deck.index)


# In[88]:

sns.factorplot('Level','Survived',x_order=['A','B','C','D','E','F'],data=titanic_df)


# 2.) Did having a family member increase the odds of surviving the crash?

# In[89]:

sns.factorplot('Alone','Survived',data=titanic_df)


# In[ ]:



