# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:57:21 2021

This is mostly a everything-goes-tutorial
a bunch of things that could or could not be useful
not in a particular well organized manner



@author: Alessio Giuseppe Ferraioli
"""


#%%   OPEN THE DATAFRAME FROM FILE

#cd D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj

import pickle
with open('df_annotated_2021-01-26.pkl','rb') as f:
    df=pickle.load(f)

#%% BIG RECAP OF WHAT YOU CAN DO WITH PANDAS DATAFRAMES

"""

Recap on how you should keep your data by Giampieri:
    
A tidy dataframe is defined as having the following properties:

- a table represent a single set of measurements
- every measured variable is represented by a column, and contains a single,
  well defined measurement
- every observed unit is represented by a row

The tidy format is brilliant, especially for long term storage and to keep
the metadata about the measurements, but some analysis might need the data 
to be transformed in non-tidy formats (for example to evaluate differences 
                                       in various time points)

For this reason, all the libraries that manage dataframe have a strong focus
on data transformation, reshaping the data from one form to the other.
This allows to easily obtain the best structure for the analysis we want 
to do without sacrificing the data quality in long term storage.

"""


import pandas as pd 
import numpy as np

"""
Pandas DataFrame holds a single table made out of various Series, 
that represent each column, all sharing the same index

let's see some basic things we can do with a dataframe
"""

# this returns general info on the dataframe, such as list of columns and types
df.info()          

# returns the first 5 rows of the dataframe
df.head()

# returns the number of rows of the dataframe
len(df)

# returns a list of column names
df.columns

# returns the number of columns of the dataframe
len(df.columns)

# returns the single column identified by the name 'id' as an array
df['id']

# the last one is equivalent to:
df.id

# returns the first 5 element of the single column identified by the name 'id' 
df['id'].head()

# returns the 0-th element of the column 'id'
df['id'][0]
    
# each Serie (=column of the DataFrame) works like a numpy array, so we can use
# array operation on them. for example:
# returns the first 5 element of the single column identified by the name 'id' each one times 100
df['id'].head() * 100

# or for example we can sum two columns as they were two arrays:
# returns an array in which each element is the sum of the two corresponding elements of the 'id' and 'user.id' columns
df['id'] + df['user.id']

# we can perform boolean operation as we do with numpy array, for example
# this returns all the element in the column 'id' greater than 5000
df['id'][df['id']>5000]   #n.b.: it gives error because the variables are strings, i should convert it to int and then it would work but i don't want to complicate the view for the sake of explanation simplicity

# this deletes a column from the dataframe, so the df dataframe would be without the 'id' column
#del df['id']      wrote it as a comment to not accidentally run this line 

# this returns only the columns of the selected dtype (in this case 'datetime64[ns, Europe/Rome]'). n.b. it shows only the first 5 rows because of .head()
df.select_dtypes(include='datetime64[ns, Europe/Rome]').head()

# this returns a dataframe in which the values are ordered with ascending order wrt 'created_at' date
df.sort_values('created_at').head()

# this returns a dataframe in which the values are ordered with descending order wrt 'created_at' date
df.sort_values('created_at', ascending=False).head()
# n.b.: these two do not modify the original dataframe, they return a new one. if i want to not return 
# anything and simply modify the input dataframe, i should write : inplace = True  (default is False)

# this returns only the part of the dataframe that satisfies the condition in brackets,in this case
df.query('id>5000')   #n.b.: id elements are strings, so i should convert it in int

# this returns a boolean array which is true if the date is greater than the one indicated 
selection_after_2016 = df['created_at']>='2017-01-01 00:00:00+02:00'

# i can use this selection mask to select only the part of the dataframe that satisfy the condition (date after 2016)
df[selection_after_2016]
# this is useful to access only a part of the dataframe without modifying it or creating a new dataframe


# useful string functions:

# returns a dataframe in which in the 'id' column the strings are splitted in the points where
# the separator is found (here it's '$', default is white space); each string is converted in
# a list of the splitted words
df['id'].str.split('$').str[-1]

# returns a dataframe in which in the 'id' column for each string, each ',' has been
# replaced with a ''
df['id'].str.replace(',', '')

# returns a dataframe in which in the 'id' column the elements have been converted in 
# the specified type, in this case np.int64
df['id'].astype(np.int64)

# Giampieri: Sometimes, one needs to use some violence...
# this converts the type but if it enconuters a value error, instead of converting it, simply
# it prints the value. in this case if we have some value that cannot be converted in int64 it
# will be simply printed without assigniging it the new type
try:
    df['id'].astype(int)
except ValueError as e:
    print(e)

''' for more complex manipulation of data (modifying the type etc) you could use
regex (regular expressions)
'''


# Generate descriptive statistics. 
# df.describe()   i put it as a comment to not run it accidentally - it takes A LOT of time!


#counting how many unique values in a column:
df['id'].nunique()  # returns how many different ids are in the df


#%%  VISUALIZATION OF DATAFRAMES 

import matplotlib.pyplot as plt

#scatter plot of two columns
plt.scatter('id', 'user.id', data=df)
#or equivalently
df.plot.scatter('id', 'user.id')

# line plot
plt.plot('id', 'user.id', data=df)


#%% PROPER VISUALIZATION OF DATAFRAMES WITH SEABORN

import seaborn

from matplotlib import style
#print(sorted(style.available))   # to show available styles
style.use('default')    # select a style for the plots


#i can now plot different columns of the dataframe wrt each other
seaborn.lmplot('id','user_id', data=df,hue='annotation')     #not that i need this
'''
https://unibodifabiophysics.github.io/programmingCourseDIFA/Lesson_09_DataFrame_and_Pandas.slides.html#/97
for more
'''


#%%
#%%
#%%  SELECTION MASK TO SELECT ONLY ANNOTATED TWEETS

''' i want to build a selection mask to get only the part of the dataframe
for which we have annotation
'''

selection_annotation = np.zeros(len(df.annotation), dtype=bool)
for i in range(len(df.annotation)):
    if type(df.annotation[i]) == float :       # if it is float it means it's a NaN, so it's a non-annotated tweet. annotated tweet have a string "Neutral"/"ProVax"/"AntiVax" in the "annotation" column
        selection_annotation[i] = False
    else:
            selection_annotation[i] = True

# now i can use this selection_annotation to only have the annotated data
# df[selection_annotation]  gives a dataframe in which we only have the annotated tweets

plt.hist('annotation', data=df[selection_annotation])








#%%%    now let's build some graph objects (networks) from this dataframe

import networkx as nx

"""
Costruire un network diretto e pesato con link da utente A a utente B 
quando A retwitta un messaggio di B. 
Quindi per fare questo vi bastano le colonne “user.id” e “retweeted_user.id”.
Potete anche provare a tenere tutti i link con peso >=2
per togliere un po’ di rumore.
"""


"""
so nodes are all the unique users in user.id and in retweeted_status_user.id
there is a link from user A to user B if user A retweeted user B (and viceversa)
the weight of the link is the number of retweets

each tweet is a directed (weight one) link, so by using the two columns 
user.id and retweeted_status_user.id i could build the directed weighted graph.

copied from: https://stackoverflow.com/questions/41646735/how-to-created-a-weighted-directed-graph-from-edge-list-in-networkx

"""


from collections import Counter


"""
i think i shoudl first of all build a selection mask to select the tweet which
retweeted another tweet and separate them from the others
problem: if 
"""

# -------------------------------------
# LET'S REMOVE TWEETS THAT DON'T RETWEET ANYONE

# this selection is True if the tweet retweeted another tweet, otherwise False
selection_retweets = np.zeros(len(df['retweeted_status.user.id']), dtype=bool)
indices_retweets = []
for i in range(len(df)):
    if type(df['retweeted_status.user.id'][i]) == float :       # if it is float it means it's a NaN, so it's a non-annotated tweet. annotated tweet have a string "Neutral"/"ProVax"/"AntiVax" in the "annotation" column
        selection_retweets[i] = False
    else:
            selection_retweets[i] = True
            indices_retweets.append(i)

# now i can access only the tweets in the dataframe that retweet another tweet
# by using df[selection_retweets]
# the indices_retweets variable stores only the indices of the dataframe for which
# we have a retweet   (MAYBE I SHOULD WRITE A FUNCTION FOR THIS) 

# this is to check if everything went in the correct way
if indices_retweets != len(df[selection_retweets]):
    print("ERROR: indices length does not match df[selection] length")
    

# -----------------------------------------
#  LET'S COLLECT ALL THE EDGES

# now i can build a vector edges of edges
edges = []
for i in indices_retweets:
    edges.append((df['user.id'][i],
               df['retweeted_status.user.id'][i]))


#---------------------------------------------
# THIS TO BUILD THE COMPLETE GRAPH (WITH ALL LINKS)

# i build a directed weighted graph with all these edges
g = nx.DiGraph((x, y, {'weight': w}) for (x, y), w in Counter(edges).items())


#-------------------------------------------------
# THIS TO BUILD THE "DENOISE" GRAPH WITH ONLY EDGES WITH WEIGHT >=2

# i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  
edges_w_weights = Counter(edges).items()

edges_w_weights_denoise = []
for (x,y), w in edges_w_weights:
    if w>=2:
        edges_w_weights_denoise.append(((x,y), w))
        
# i build the directed weighted graph object with only edges with weight>=2    
g_denoise = nx.DiGraph(edges_w_weights_denoise)   


    
    # we should first of all remove the tweets that don't retweet anything
    indices_retweets = Select('retweeted_status.user.id')[0] # we save only the indices of the dataframe that correspond to a tweet that retweets another
    
    indices_user = Select(indices_user_name)
    
    indices = list(set(indices_retweets) & set(indices_user))
        
    edges = []
    for i in indices:
        edges.append((df[columns_names[0]][i],
                      df[columns_names[1]][i]))

    if denoise == False:  #if i don't have to discard some edges, i simply build the graph
        g = nx.DiGraph((x, y, {'weight': w}) for (x, y), w in Counter(edges).items())

    else:       # denoise == True means i only save edges with weight >=2
        
        edges_w_weights = Counter(edges).items()   # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  

        edges_w_weights_denoise = []
        for (x,y), w in edges_w_weights:      # i save in edges_w_weights_denoise only edges with weight>=2
            if w>=2:
                edges_w_weights_denoise.append(((x,y), w))

        g = nx.DiGraph(edges_w_weights_denoise)      # i build the graph with the edges with weight >=2
    
    
    return g
     

    
#%% cerchiamo di capirci qualcosa in più

"""

cerchimao di capire se veramente i tweet annotati sono tutti senza retweet
come posso fare? 
.......

potrei prendere un attimo il dataframe dei soli tweet annotati (non è molto grande)
e in questo controllare se ci stanno non NaN nel "retweeted_user.id"


"""

indices_annotation, selection_mask_annotation = Select(df, 'annotation')



indices_non_nan = []
for i in indices_annotation: 
    if type(df[selection_mask_annotation]['retweeted_status.user.id'][i]) != float:
        indices_non_nan.append(i)



'''
ok ho appurato che tutti i tweet annotati non sono retweet. 
fammi vedere che sovrapposizione ci sta tra 'annotation' (annotazione del tweet)
e 'user_annotation'' (annotazione dell'autore del tweet)

'''


indices_user_annotation, selection_mask_user_annotation = Select(df, 'user_annotation')


indices_non_nan = []
for i in indices_user_annotation: 
    if type(df['retweeted_status.user.id'][i]) != float:
        indices_non_nan.append(i)



#%%


# this is a network with twitter users as nodes; there is a (weighted directed) edge 
# between two nodes if user A retweeted user B
# all the weights < 2 have been discarded
G_user_annotation_denoise = Build_Weighted_DiGraph(df, 
                                                   ('user.id',
                                                    'retweeted_status.id'),
                                                   'user_annotation')


# i would like to include the "user annotation" as an attribute of the nodes 







#%%%%%

nx.draw(G_user_annotation_denoise)

#%%%%




edges = []
for i in range(len(df_user_annotation_retweet)):
    edges.append((df_user_annotation_retweet['user.id'].iloc[i],
                  df_user_annotation_retweet['retweeted_status.user.id'].iloc[i]))

edges_w_weights = Counter(edges).items()   # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  

edges_w_weights_denoise = []
for (x,y), w in edges_w_weights:      # i save in edges_w_weights_denoise only edges with weight>=2
    if w>=2:
        edges_w_weights_denoise.append(((x,y), w))




G_user_annotation_denoise = nx.DiGraph(edges_w_weights_denoise)      # i build the graph with the edges with weight >=2




#%%%%%%%%%%%%


    
source_name = 'user.id'
target_name = 'retweeted_status.user.id'
selection_name = 'user_annotation'
weight_minimum = 220

df_selected = df[df[selection_name].notna()]

df_selected_retweet = df_selected[df_selected['retweeted_status.user.id'].notna()]


edges = []
for i in range(len(df_selected_retweet)):
    edges.append((df_selected_retweet[source_name].iloc[i],
                  df_selected_retweet[target_name].iloc[i],
                  df_selected_retweet['user_annotation'].iloc[i]))


edges_w_weights = Counter(edges).items()   # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  

edges_w_weights_denoise = []
for (x,y,z), w in edges_w_weights:      # i save in edges_w_weights_denoise only edges with weight>=2
    if w>=weight_minimum:
        edges_w_weights_denoise.append((x, y, w, z))

# NOTE that edges_w_weights has this structure "NODE FROM, NODE TO, WEIGHT, ATTRIBUTE(s)"


weighted_edgelist = []
attributes = []
for x in edges_w_weights_denoise:
    weighted_edgelist.append((x[0],x[1],x[2]))
    attributes.append(x[3])

G = nx.DiGraph()      # i build the graph with the edges with weight >=2
G.add_weighted_edges_from(weighted_edgelist, attr=attributes)


pos = nx.circular_layout(G)
nx.draw_networkx(G,pos,font_color='k',node_size=500, edge_color='b', alpha=0.5)




#%%  i would like to see how many different users we have

set_user_id = set(df_selected_retweet['user.id'])
set_retweeted_user_id = set(df_selected_retweet['retweeted_status.user.id'])
list_users = [*df_selected_retweet['user.id'], *df_selected_retweet['retweeted_status.user.id']]
set_users = set(list_users)



#%% useful things 


nx.number_of_selfloops(G)
G.degree
G.in_degree
G.out_degree

nx.nodes_with_selfloops(G)
list(nx.nodes_with_selfloops(G))

G.remove_edges_from(nx.selfloop_edges(G))  # remove self loop edges

nx.isolates(G)    #nodes not connected with anything
list(nx.isolates(G))   #lists the isolated

G.remove_nodes_from(list(nx.isolates(G)))




def Remove_Those_Stupid_Motherfuckers_Who_Only_Retweet_Their_Own_Tweets(G):
    
    G.remove_edges_from(nx.selfloop_edges(G))  # remove self loop edges
    G.remove_nodes_from(list(nx.isolates(G)))  # remove isolated nodes
    
    return G


#%%

degrees = nx.degree_centrality(G)
betweeness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)




#%%



def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

plot_degree_dist(G)



#%%%  now i want to check if user_annotation is consistent
# i.e. if a user is once tagged eg novax then it will be always tagged novax


df_selected = df[df['user_annotation'].notna()]
df_selected_retweet = df_selected[df_selected['retweeted_status.user.id'].notna()]

black_list=[]
for i in range(len(df_selected_retweet)):
    for j in range((len(df_selected_retweet)) - i):
        if df_selected_retweet['user.id'].iloc[i] == df_selected_retweet['user.id'].iloc[j]:  #if they are the same user
            if df_selected_retweet['user_annotation'].iloc[i] != df_selected_retweet['user_annotation'].iloc[j]:
                black_list.append(df_selected_retweet['user.id'].iloc[i])
                i = i+1
        







#%%%%%%
#%%%%%
#%%%%%%%%%
#%%%%%%%%

#%%%%%%
#%%%%%
#%%%%%%%%%
#%%%%%%%%

#%%%%%%
#%%%%%
#%%%%%%%%%
#%%%%%%%%


#%%
clustergraph=communities.cluster_graph()
giant = communities.giant()

small_communities_indices = []




ig.plot(giant,"D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_test_giant1.png", **visual_style)




