# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:26:15 2021



@author: Alessio Giuseppe Ferraioli 
"""

#%%              OPEN THE DATAFRAME FROM FILE

#cd D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj

import pickle
with open('df_annotated_2021-01-26.pkl','rb') as f:
    df=pickle.load(f)
    
    
#%%                  IMPORTS

import pandas as pd 
import numpy as np
import networkx as nx
from collections import Counter


#%%              Functions useful to build the network from the dataframe


def Select(df, column_name):


    '''
    
    imports required: pandas as pd
    
    This function select a subset of the indices of a dataframe df, discarding 
    the indices for which we have a NaN in the column indicated by column_name.
    
    inputs:
        - df, pandas dataframe
        - column_name, string: the name of the column of the dataframe in which
          we check for NaNs.
          

    outputs:
        - indices, list: list of int values of the indices of the dataframe for
          which we don't have a NaN in the column indicated by column_name
        - selection_mask, bool list: list of len(df) which is True for the indices
          in indices and false otherwise
    
    '''
    

    indices = []
    selection_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if type(df[column_name][i]) == float :       # if it is float it means it's a NaN, so it's a non-annotated tweet. annotated tweet have a string "Neutral"/"ProVax"/"AntiVax" in the "annotation" column
            selection_mask[i] = False
        else:
            selection_mask[i] = True
            indices.append(i)


    return (indices, selection_mask)


def Build_Weighted_DiGraph(df, columns_names, indices_user_name='user.id', denoise=True):
    
    
    '''
    
    imports required: pandas as pd, networkx as nx, numpy as np, from collections Counter
    
    This function builds a Weighted Directed Graph from the dataframe df.
    
    inputs: 
        - df, pandas DataFrame: each row is a (weight one) edge of the network
        - columns_names, (str,str): the names of the two columns of df that we
          use as the edge list. i.e. the two columns x and y such that x_i and
          y_j are nodes and there is a (weight one) link from x_i to y_j if i=j
        - indices_user_name, string: "name of the user defined column from 
          which to find indices". It specifies a column name that is used to
          select with the Select function only the indices for which that
          column has not a NaN. 
          Default is 'user.id', which has 0 NaNs, so in that case it does nothing.
        - denoise, bool: if it is True, the network will be constructed using
          only edges with weight >= 2. If it is False, all the edges will be used. 
          
    outputs:
        - G, networkx graph object.
        
    '''
    
    # we should first of all remove the tweets that don't retweet anything
    indices_retweets = Select(df, 'retweeted_status.user.id')[0]  # we save only the indices of the dataframe that correspond to a tweet that retweets another
    
    indices_user = Select(df, indices_user_name)[0]
    
    indices = list(set(indices_retweets) & set(indices_user))
        
    edges = []
    for i in indices:
        edges.append((df[columns_names[0]][i],
                      df[columns_names[1]][i]))

    if denoise == False:  #if i don't have to discard some edges, i simply build the graph
        G = nx.DiGraph((x, y, {'weight': w}) for (x, y), w in Counter(edges).items())

    else:       # denoise == True means i only save edges with weight >=2
        
        edges_w_weights = Counter(edges).items()   # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  

        edges_w_weights_denoise = []
        for (x,y), w in edges_w_weights:      # i save in edges_w_weights_denoise only edges with weight>=2
            if w>=2:
                edges_w_weights_denoise.append(((x,y), w))

        G = nx.DiGraph(edges_w_weights_denoise)      # i build the graph with the edges with weight >=2
    
    
    return G, indices


#%%

# this is a network with twitter users as nodes; there is a (weighted directed) edge 
# between two nodes if user A retweeted user B
# all the weights < 2 have been discarded
G_user_annotation_denoise, indices_user_annotation_denoise = Build_Weighted_DiGraph(df, 
                                                   ('user.id',
                                                    'retweeted_status.id'),
                                                   'user_annotation')
#%%
# setting the attribute (WARNING: this is O(n^2) in n number of nodes - if you find
# a more efficient way discard this)
for i in G_user_annotation_denoise.nodes:   #i select a node
    for j in indices_user_annotation_denoise:     # i select an index of the df
        if G_user_annotation_denoise.nodes[i] == df['user.id'][j]:         # se il nome del nodo combacia con l'user id
            if df['user_annotation'][j] == 'ProVax':
                G_user_annotation_denoise['user_annotation'][i] = 'ProVax'
            elif df['user_annotation'][j]  == 'NoVax':
                G_user_annotation_denoise['user_annotation'][i] = 'NoVax'
            elif df['user_annotation'][j] == 'Neutral':
                G_user_annotation_denoise['user_annotation'][i] = 'Neutral'
            else:
                print('error at:')
                print(i)
        
        #j++ (per uscire fuori dal loop di j, ma non posos semplicemente avanzare di 1 perché sto in indices)
        
        
        
# APPARENTLY THIS DOES NOT WORK BECAUSE I AM NOT ABLE TO ACCESS THE INDIVIDUAL NODES 
# (THE NODE NAMES THAT SHOULD BE THE USER.IDs)














