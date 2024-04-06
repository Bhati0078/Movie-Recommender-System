#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies = pd.read_csv("D:/Downloads/movie/tmdb_5000_movies.csv")
credits = pd.read_csv("D:/Downloads/movie/tmdb_5000_credits.csv")


# In[278]:


movies.head(1)


# In[279]:


credits.head(1)


# In[280]:


credits.rename(columns={'movie_id': 'id'}, inplace=True)


# In[281]:


# Merging the Dataset on the bases of ID
movie = movies.merge(credits,on='id')


# In[282]:


movie.head(1)


# In[283]:


movie.drop(columns= 'title_y',axis=1,inplace=True)


# In[284]:


movie.rename(columns={'title_x':'title'},inplace=True)


# In[285]:


movie.info()


# In[286]:


movie.shape


# In[287]:


movie_f = movie[['id','title','genres','keywords','overview','cast','crew']]
movie_f.head()


# In[288]:


movie_f.isnull().sum()


# In[289]:


movie_f.dropna(inplace=True)


# In[290]:


movie_f.duplicated().sum()


# In[291]:


movie_f.iloc[0].genres


# In[292]:


def convert(obj):
    name_list = []
    for i in ast.literal_eval(obj):
        name_list.append(i['name'])
    return name_list



# In[293]:


movie_f['genres'] = movie_f['genres'].apply(convert)


# In[294]:


movie_f.iloc[0].genres


# In[295]:


movie_f['keywords'] = movie_f['keywords'].apply(convert)


# In[296]:


movie_f.iloc[0].keywords


# In[297]:


def get_actor(obj):
    name_list = []
    counter = 0
    for i in ast.literal_eval(obj):
      if counter != 5:
        name_list.append(i['name'])
        counter +=1
      else:
        break
    return name_list



# In[298]:


movie_f['cast'] = movie_f['cast'].apply(get_actor)


# In[299]:


movie_f.iloc[0].cast


# In[300]:


movie_f['crew'][0]


# In[301]:


def get_director(obj):
    name_list = []
    for i in ast.literal_eval(obj):
      if i['job'] == 'Director':
        name_list.append(i['name'])
        break
    return name_list



# In[302]:


movie_f['Director'] = movie_f['crew'].apply(get_director)


# In[303]:


movie_f.iloc[0].Director


# In[304]:


movie_f.iloc[0].overview


# In[305]:


movie_f['overview'] = movie_f['overview'].apply(lambda x:x.split())


# In[306]:


movie_f.iloc[0].overview


# In[307]:


movie_f.head(5)


# In[308]:


movie_f['genres'] = movie_f['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movie_f['keywords'] = movie_f['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movie_f['cast'] = movie_f['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movie_f['Director'] = movie_f['Director'].apply(lambda x:[i.replace(' ','') for i in x])


# In[309]:


movie_f.head(1)


# In[310]:


movie_f['tags'] = movie_f['overview'] + movie_f['genres'] + movie_f['keywords'] + movie_f['cast'] + movie_f['Director']


# In[311]:


movie_f.head()


# In[312]:


new_movie = movie_f[['id','title','tags']]
new_movie.head()


# In[313]:


new_movie['tags'] = new_movie['tags'].apply(lambda x:' '.join(x))
new_movie.head()


# In[314]:


# Stemming aims to reduce each word to its root or base form
def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return ' '.join(y)


# In[315]:


new_movie['tags'] = new_movie['tags'].apply(stem)


# In[316]:


new_movie['tags'][0]


# In[317]:


new_movie['tags'] = new_movie['tags'].apply(lambda x:x.lower())


# In[318]:


new_movie.head(2)


# In[319]:


cv = CountVectorizer(max_features=5000,stop_words='english')


# In[320]:


vectors = cv.fit_transform(new_movie['tags']).toarray()


# In[321]:


cv.get_feature_names_out()


# In[322]:


list(cv.get_feature_names_out())


# In[323]:


similarity = cosine_similarity(vectors)


# In[324]:


def recommend(movie):
  movie_index = new_movie[new_movie['title'] == movie].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]
  for i in movie_list:
    print(new_movie.iloc[i[0]].title)



# In[325]:


recommend('Batman Begins')


# In[326]:


new_movie.iloc[778].title


# In[327]:


import pickle


# In[328]:


pickle.dump(new_movie.to_dict(),open('movie_dict.pkl','wb'))


# In[329]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




