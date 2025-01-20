# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:23:38 2021

building the graph v3 (major breakthrouhg)


@author: Utente
"""

#%%              OPEN THE DATAFRAME FROM FILE

import pickle
with open('df_annotated_2021-01-26.pkl','rb') as f:
    df=pickle.load(f)
    
    
#%%                  IMPORTS

import pandas as pd 
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt


#%%

def Build_Weighted_DiGraph(df, source_name, target_name, selection_name='user.id',
                           weight_minimum=1, remove_isolates=True, 
                           only_largest_component=True, prints=False):
        

    
    '''
   
    This function builds a Weighted Directed Graph from the dataframe df.

    imports required: pandas as pd, networkx as nx, from collections Counter
    
    
    inputs: 
        - df, pandas DataFrame: each row is a (weight one) edge of the network
        - source_name, string: name of the column whose elements are the nodes
          from which the edges start. 
        - target_name, string: name of the column whose elements are the nodes
          from which the edges end. 
        - selection_name, string: name of the column that is used to
          select a subset of the dataframe. Rows for which in the "selection_name"
          column there is a NaN will be discarded.
          Default is 'user.id', which has no NaNs, so in that case it does nothing.
        - weight_minimum, int: minimum edge weight to keep. All the edges with
          weight less than weight_minimum will be discarded.
          Default is 1, so in that case no edge is discarded.
        - remove_isolates, bool: if it's True, all the nodes not connected with
          any other node would be discarded. Default is True
        - only_largest_components, bool: if it's True, only the largest
          component of the graph will be kept. Default is True.
        - prints, bool: if it's True, additional information will be printed in
          rundown. Default is False, so no information will be printed.
          
    outputs:
        - G, networkx graph object.
        
    '''

    # i select only the subset of the dataframe i need
    df_selected = df[df[selection_name].notna()]
    df_selected_retweet = df_selected[df_selected['retweeted_status.user.id'].notna()]
    
    # i save the edges in a edges list (with additional attributes)
    edges = []
    for i in range(len(df_selected_retweet)):
        edges.append((df_selected_retweet[source_name].iloc[i],
                      df_selected_retweet[target_name].iloc[i],
                      df_selected_retweet['user_annotation'].iloc[i]))
    
    
     # i save in a vector the edges with weights as :  ((x,y),w)   x y nodes, w weight  
    edges_w_weights = Counter(edges).items()  
    
    
    
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
    G.add_weighted_edges_from(weighted_edgelist)
    
    
    # pos = nx.circular_layout(G)
    # nx.draw_networkx(G,pos,font_color='k',node_size=500, edge_color='b', alpha=0.5)
    
    if remove_isolates == True:
        #Removing Isolated Users that only retweet themselves
        G.remove_edges_from(nx.selfloop_edges(G))  # remove self loop edges
        G.remove_nodes_from(list(nx.isolates(G)))  # remove isolated nodes
        
    if only_largest_component == True:
        G = max(nx.weakly_connected_components(G), key=len)

    return G


#%%


G = Build_Weighted_DiGraph(df, 
                           source_name='user.id',
                           target_name='retweeted_status.user.id', 
                           selection_name='user_annotation',
                           weight_minimum=2,
                           remove_isolates=True, 
                           only_largest_component=True)

        