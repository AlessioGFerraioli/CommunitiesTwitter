# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:26:47 2021


i try to build the graph by creating the empty graph object and then adding
edges with add_edges_from instead of using the function DiGraph(edgelist) 
to see if it's better this way

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

#%%




def Build_Weighted_DiGraph(df, source_name, target_name, selection_name='user.id', weight_minimum=1, prints=False):
    
    
    '''
    
    imports required: pandas as pd, networkx as nx, numpy as np, from collections Counter
    
    This function builds a Weighted Directed Graph from the dataframe df.
    
    inputs: 
        - df, pandas DataFrame: each row is a (weight one) edge of the network
        - source_name/target_name, str: the name of the two columns of df that we
          use as the edge list. i.e. the two columns x and y such that x_i and
          y_j are nodes and there is a (weight one) link from x_i to y_j if i=j
        - indices_user_name, string: "name of the user defined column from 
          which to find indices". It specifies a column name that is used to
          select with the Select function only the indices for which that
          column has not a NaN. 
          Default is 'user.id', which has 0 NaNs, so in that case it does nothing.
        - weight_minimum, int: all the edges with weight < weight_minimum will be
          discarded and would not form the network. Default is 1, so no edge will
          be discarded.
        - prints, bool: if it's True the function will print some additional info,
          if it's False it will not print anything. Default is False. 
          
    outputs:
        - G, networkx graph object.
        
    '''
    
    # we should first of all remove the tweets that don't retweet anything
    df_selected = df[df[selection_name].notna()]

    df_selected_retweet = df_selected[df_selected['retweeted_status.user.id'].notna()]


    edges = []
    for i in range(len(df_selected_retweet)):
        edges.append((df_selected_retweet[source_name].iloc[i],
                      df_selected_retweet[target_name].iloc[i]))

    if prints == True:
        print('total number of edges:')
        print(len(edges))
    
    if weight_minimum < 2:  #if i don't have to discard some edges, i simply build the graph
        G = nx.DiGraph((x, y, {'weight': w}) for (x, y), w in Counter(edges).items())      # i use Counter to count how many times an edge is repeated in the edges list; it returns a 3-tuple list as : ((x,y), w), x y nodes, w weight

    else:       # if weight_minimum >=2 it means i have to discard some edges
        
        edges_w_weights = Counter(edges).items()   # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  

        edges_w_weights_denoise = []
        for (x,y), w in edges_w_weights:      # i save in edges_w_weights_denoise only edges with weight>=2
            if w>=weight_minimum:
                edges_w_weights_denoise.append(((x,y), w))

        G = nx.Graph()      # create empty graph object
        G.add_edges_from(edges_w_weights_denoise)   # add the edges with weights
        
    
    if prints == True:
        print('number of edges with weight:')
        print(len(edges_w_weights))

    
    return G


#%%



G_user_annotation_denoise = Build_Weighted_DiGraph(df, 'user.id',
                                                    'retweeted_status.user.id',
                                                    'user_annotation', 2)




