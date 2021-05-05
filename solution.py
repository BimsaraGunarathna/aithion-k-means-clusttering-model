#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# In[2]:


data=pd.read_csv("data.csv")


# In[3]:


data["id"] = data.index + 1
#index+1- id column


# In[4]:


del data["country"]


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[8]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[9]:


kmeans.inertia_


# In[10]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[11]:


# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_clusters = 3, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)


# In[12]:


frame = pd.DataFrame(data_scaled)
frame["id"] = frame.index + 1
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[13]:


print(frame['cluster'])


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


pred


# In[16]:


u_labels = np.unique(pred)


# In[17]:


for i in u_labels:
    plt.scatter(data_scaled[pred == i , 0] , data_scaled[pred == i , 1] , label = i)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[18]:


print (frame[frame.cluster == 1].shape[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


OriginalData=pd.read_csv("data.csv")


# In[ ]:





# In[20]:


data_o=pd.read_csv("data.csv")


# In[21]:


data_o["id"] = data_o.index + 1


# In[22]:


df3=pd.merge(data_o,frame,on="id")


# In[23]:


df3


# In[24]:


cluster0=df3[df3["cluster"] == 0]


# In[25]:


cluster0


# In[26]:


mean_0 = cluster0["life_expectancy"].mean()


# In[27]:


mean_0


# In[28]:


cluster1 = df3[df3["cluster"] == 1]


# In[29]:


cluster1


# In[30]:


mean_1 = cluster1["life_expectancy"].mean()


# In[31]:


mean_1


# In[32]:


cluster2=df3[df3["cluster"] == 2]


# In[33]:


cluster2


# In[34]:


mean_2 = cluster2["life_expectancy"].mean()


# In[35]:


mean_2


# In[36]:


if mean_0 < mean_1 and mean_0 < mean_2 :
    u_d_cluster = cluster0
if mean_1 < mean_0 and mean_1 < mean_2 :
    u_d_cluster = cluster1
if mean_2 < mean_0 and mean_2 < mean_1 :
    u_d_cluster = cluster2


# In[37]:


country_column = u_d_cluster.loc[:,'country']


# In[38]:


countries = country_column.values


# In[39]:


countries


# In[40]:


textfile= open("outcomes.txt","w+")
textfile.write('No. of under-developing countries : ')
NoOfCountries=u_d_cluster.shape[0]
textfile.write(str(NoOfCountries))
textfile.write('\n')


# In[41]:


textfile.write('Under-developing countries: ')
textfile.write('[')


# In[42]:


with open('outcomes.txt', 'a') as filehandle:
    #textfile.write (', \'' .join(countries).'\'')
    #textfile.write (', '.join(f'"{w}"' for w in countries))
    textfile.write ("'"+"','".join(countries)+"'")
    #for listitem in countries:
       #textfile.write (',' .join(countries))
       #textfile.write( "'" + listitem +"', ") 
textfile.write("]")


# In[43]:


textfile.write('\n')
textfile.write('The no. of vaccines a country would receive: ')
NoOfVaccines = 20000000
OneCountry = NoOfVaccines/NoOfCountries
OneCountryR=round(OneCountry)
textfile.write(str(OneCountryR))
textfile.write('\n')


# In[44]:


textfile.close()

