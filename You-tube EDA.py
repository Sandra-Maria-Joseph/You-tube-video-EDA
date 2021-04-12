#!/usr/bin/env python
# coding: utf-8

# 
# 
# ### Exploratory Data Analysis on YouTube Dataset

# ![image.png](attachment:image.png)

# #### Importing Packages

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import json
from matplotlib import cm
from datetime import datetime
import glob
import re


# In[3]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ### Importing First Dataset

# In[4]:


data = pd.read_csv(r"E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_1\data\CAvideos.csv")
data.head()


# ###### Variable Description
# 1)video_id: unique ID number to each video
# 
# 2)trending_date: date of the video was trend
# 
# 3)title: video title
# 
# 4)channel_title: channel title
# 
# 5)category_id: ID number of a video's region
# 
# 6)publish_time: time of the video is published
# 
# 7)tags: tags about the video
# 
# 8)views: number of views of a video
# 
# 9)likes: number of likes of a video
# 
# 10)dislikes: number of dislikes of a video
# 
# 11)comment_count: number of comments about a video
# 
# 12)thumbnail_link: thumbnail link
# 
# 13)comments_disabled: comments are disabled(True) or comments are available(False)
# 
# 14)ratings_disabled: ratings are disabled(True) or ratings are available(False)
# 
# 15)video_error_or_removed: video gives error/removed(True) or video is available(False)
# 
# 16)description: descriptions under the video

# In[5]:


data.shape


# In[6]:


data.info()


# ###### Here, we can understand from this output that; our data has 16 columns and 40949 rows.
# 
# bool(3): comments_disabled, ratings_disabled, video_error_or_removed
# 
# int64(5): category_id, views, likes, dislikes, comment_count
# 
# object(8): video_id, trending_date, title, channel_title, publish_time, tags, thumbnail_link, description

# In[7]:


data.describe()


# ### Importing Second Dataset

# In[8]:


with open(r'E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_1\data\CA_category_id.json') as json_file:
    datay = json.load(json_file)

datay['items'][3]['id']
for i in range(len(datay['items'])):
    print(datay['items'][i]['snippet']['title'])


# ## Data Cleaning

# #### Cleaning the publish_time and trending_date columns

# In[9]:


dateparse1 = lambda x: pd.datetime.strptime(x, '%y.%d.%m')
data['trending_date'] = data.trending_date.apply(dateparse1)
dateparse2 = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
data['publish_time'] = data.publish_time.apply(dateparse2)


# In[10]:


data.insert(4, 'publish_date', data.publish_time.dt.date)
data['publish_time'] = data.publish_time.dt.time
data.head()


# In[11]:


full_df = data.reset_index().sort_values('trending_date').set_index('video_id')
df = data.reset_index().sort_values('trending_date').drop_duplicates('video_id', keep='last').set_index('video_id')
df.head()


# ##### There are two sets of data:
# df, which removed duplicated data by keeping only last entry since it contains the latest stats
# 
# full_df, which contains all entries.

# ### Joining Category ID

# In[12]:


idP = []

for i in range(30):
    idP.append(datay['items'][i]['id'])
    


# In[13]:


title = []
for i in range(30):
    title.append(datay['items'][i]['snippet']['title'])


# In[14]:


d = {'id': idP, 'title': title}

df2=pd.DataFrame(d)
df2


# In[15]:


df2.id = df2.id.astype(int)
df2.dtypes
df3 = pd.merge(df,df2, left_on='category_id', right_on='id')
df3


# In[16]:


df4=df3.insert(4, 'category', df3.title_y)
df3.drop(columns='title_y')
df = df3
df = df.set_index('trending_date').drop(columns=['title_y'])
df = df.drop(columns=['id'])


# In[17]:


df.head(5)


# In[18]:


df.category.unique()


# ## Exploratory Data Analysis

# ### What kinds of videos make it to the trending list worldwide?

# In[19]:


fig = px.histogram(df, x=df['category'])

fig.update_layout(title = {'text':'Number of videos sorted by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Count',
                 template='seaborn')

fig.show()


# ####  Observation
# The videos Belonging to the Entertainment Category is more trending, followed by News and Politics, and People and Blogs.

# ### Normalization

# In[20]:


categorys = df["category"].value_counts(normalize=True).reset_index()
# rename columns names
categorys.rename(columns={"index":"Category","category":"count"},inplace=True)
# we use style background ¡
categorys.style.background_gradient(cmap='mako_r')


# ## Corelation matrix

# In[21]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap='YlGn_r')
plt.show()


# In[22]:


sns.regplot(x='views',y='likes',data=df,color='red').set_title('Likes v/s Views')


# In[23]:


sns.regplot(x='comment_count',y='views',data=df,color='green').set_title('Comment v/s Views')


# In[24]:


sns.regplot(x='likes',y='comment_count',data=df,color='magenta').set_title('Likes v/s Commment')


# # What categories do the top 100 trending videos (by views) belong to?

# In[25]:


m=df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count()[['views']]
p=m.sort_values(by='views',ascending=False)
p.style.background_gradient(cmap='mako_r')


# In[26]:


df_val = df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count().title_x
df_index = df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count().index
#fig = px.pie(df_val, values=df_index, names=df_index, title='Reigon wise Profit Earned')
#fig.show()
plt.pie(df_val, labels=df_index, radius = 2, autopct = '%0.2f%%')
plt.show()


# #### Observation:
# The most popular (most viewed) trending videos are predominantly music videos.
# The second most popular are entertainment videos. However, this is also because there are far fewer trending music videos than there are of other category as music videos stay trending for longer, as we shall see below.

# ### Which Category Gets Maximum Likes?

# In[27]:


fig = px.histogram(df, x=df['category'],y=df['likes'],color=df['category'],color_discrete_sequence=['yellow'])

fig.update_layout(title = {'text':'Category v/s likes',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Count_of_likes',
                 template='seaborn')

fig.show()


# ### Which Category Gets Maximum Comments?

# In[28]:


fig = px.histogram(df, x=df['category'],y=df['comment_count'],color=df['category'],color_discrete_sequence=['green'])

fig.update_layout(title = {'text':'Category v/s Comment',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Count_of_comments',
                 template='seaborn')

fig.show()


# ## Which category gets maximum Dislikes?

# In[29]:


fig = px.histogram(df, x=df['category'],y=df['dislikes'],color=df['category'],color_discrete_sequence=['magenta'])

fig.update_layout(title = {'text':'Category v/s Dislikes',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Count_of_dislikes',
                 template='seaborn')

fig.show()


# ### Checking Skewness

# In[30]:


print("Views quantiles")
print(df['views'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Likes quantiles')
print(df['likes'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Disikes quantiles')
print(df['dislikes'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Comments quantiles')
print(df['comment_count'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')


# The data is extremely skewed with a big gap between the first 75% and the last 25%
# 
# * 75% of views are less than 7L and the last 25% goes upto 2.5 B
# 
# * 75% of likes are less than 40 K and the last 25% goes upto 800 K
# 
# * 75% of dislikes are less than 700 and the last 25% goes upto 17 K
# 
# * 75% of comments are less than 2K and the least 25% goes upto 39 K

# ### Log Transform
# We will logarithmize the data to fix the above mentioned problem as the they will play a big role in a boxplot

# In[31]:


df['views_log'] = np.log(df['views'] + 1)
df['likes_log'] = np.log(df['likes'] + 1)
df['dislikes_log'] = np.log(df['dislikes'] + 1)
df['comments_log'] = np.log(df['comment_count'] + 1)


# ## Views distribution by category

# In[32]:


fig = px.box(df, x=df['category'], y=df['views_log'])

fig.update_layout(title = {'text':'Views distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='views',
                 template='seaborn',
                 )

fig.show()


# ## Likes distribution by category

# In[33]:


fig = px.box(df, x=df['category'], y=df['likes_log'],color=df['category'],
                color_discrete_sequence=[ "goldenrod"])

fig.update_layout(title = {'text':'Likes distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='likes_log',
                 template='seaborn')

fig.show()


# ### Dislikes distribution by category¶

# In[34]:


fig = px.box(df, x=df['category'], y=df['dislikes_log'],color=df['category'],
                color_discrete_sequence=[ 'darkturquoise'])

fig.update_layout(title = {'text':'Dislikes distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='dislikes_log',
                 template='seaborn')

fig.show()


# ### Comments distribution by category

# In[35]:


fig = px.box(df, x=df['category'], y=df['comments_log'],color=df['category'],
                color_discrete_sequence=[ 'mediumorchid'])

fig.update_layout(title = {'text':'Comments distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='comments_log',
                 template='seaborn')

fig.show()


# Music and Science&Technology have have the highest engagement in all the quartiles
# 
# - Fourth quartile: Music is the highest in all 4 metrics
# 
# - Third quartile: Music is the highest in all 4 metrics
# 
# - Second quartile: Gaming and music are the highest in all 4 metrics
# 
# - First quartile: Music , comedy and gaming are the highest in all 4 metrics
# 
# - Min value: Education is the highest in all 4 metrics
# 

# ### Visualize top 10 by feature

# In[36]:


def top_10(df, col, num=10):
    sort_df = df.sort_values(col, ascending=False).iloc[:num]
    
    fig = px.bar(sort_df, x=sort_df['title_x'], y=sort_df[col],color=sort_df['title_x'],
                color_discrete_sequence=["mediumaquamarine",'yellowgreen','mediumorchid','rebeccapurple','slateblue',' papayawhip','lightsalmon',"chocolate",
                                        ])
    
    labels = []
    for item in sort_df['title_x']:
        labels.append(item[:10] + '...')
        
    fig.update_layout(title = {'text':'Top {} videos with the highest {}'.format(num, col),
                           'y':0.95,
                           'x':0.4,
                            'xanchor':'center',
                            'yanchor':'top'},
                 xaxis_title='',
                 yaxis_title=col,
                     xaxis = dict(ticktext=labels))
  
    fig.show()
    
    return sort_df[[ 'title_x', 'channel_title','category', col]]


# ### Top 10 videos with the highest views

# In[37]:


top_10(df, 'views', 10)


# ### Top 10 videos with the highest likes

# In[38]:


top_10(df, 'likes')


# ### Top 10 videos with the highest Dislikes

# In[39]:


top_10(df, 'dislikes')


# In[40]:


top_10(df, 'comment_count')


# 3 entries in all the 4 features, shows that videos with high views are also prone to higher engagement
# 
# 6 common entries in top 10 views and likes, shows a high correlation between views and likes
# 
# 3 common entries in top 10 comment_count and dislikes, shows a correlation between comments and dislikes
# 
# 3 common entries in top 10 comment_count and likes, shows a correlation between comments and likes
# 
# comment_count has a correlation with views, likes and dislikes

# ## CONCLUSION

# ###### *The videos Belonging to the Entertainment Category is more trending, followed by News and Politics, and People and Blogs.

# ##### * There is a high corelation between likes and comments, views and likes, and dislikes_log and likes_log

# #### *The most popular (most viewed) trending videos are predominantly music videos. The second most popular are entertainment videos.

# ### In order to predict the Category of YouTube Videos,we can Use the Classification Techniques.
# 

# ##### Classifiers that can be used:
# #### *Multinomial Naive Bayes
# #### *Random Forest
# #### *Support Vector
# #### *K Neighbors
# #### *Decision Tree

# In[41]:


import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[42]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[43]:


vector = CountVectorizer()
counts = vector.fit_transform(df['title_x'].values)


# In[44]:


NB_Model = MultinomialNB()
RFC_Model = RandomForestClassifier()
SVC_Model = SVC()
KNC_Model = KNeighborsClassifier()
DTC_Model = DecisionTreeClassifier()


# In[46]:


output = df['category_id'].values


# In[48]:


NB_Model.fit(counts,output)


# In[49]:


RFC_Model.fit(counts,output)


# In[ ]:




