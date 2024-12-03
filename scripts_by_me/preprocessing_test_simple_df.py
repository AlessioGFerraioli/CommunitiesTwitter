# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:42:35 2022

This is just a simple test program to run with a simple test
dataframe i manually wrote just to check if the preprocessing
we do on the actual dataframe does what we think it does

I'm saving this file just in case I need to test something
with a simple test dataframe

@author: Utente
"""

# i try to see if the actual preprocessing is done correctly by creating
# a test small dataframe and see actually what it does

import pandas as pd   #i will not use it, maybe?

#%% creating a simple test dataframe


df = {'user.id': ['10', '20', '40', '50', '10'],
        'user_annotation': ['AntiVax', 'ProVax', float('nan'), 'AntiVax', 'AntiVax'],
        'retweeted_status.id': ['99', float('nan'), '44', '77', '11'],
        'retweeted_status.user.id': ['20', '40', '20', '30', '20' ]  
        }

#%% visualizing
df = pd.DataFrame(df)

print (df)



#%% preprocessing:

 
# i remove from my dataframe the rows for which user_annotation is nan
 # (we only want to keep the user with a user_annotation)
df = df.loc[~df["user_annotation"].isna()]
 
 # i remove from my dataframe the rows for which "retweeted_status.id" is nan
 # (we only want to keep the tweets that are retweets)
df = df.loc[~df["retweeted_status.id"].isna()]
 
 # we count how many times user A retweeted user B and store the values in a "weight" attribute
df['weight'] = df.groupby(['user.id', 'retweeted_status.user.id'])['user.id'].transform('size')
 
 
 #i remove all the edges with weight<2
weight_cut = 0
df = df.loc[df["weight"] >= weight_cut]


#%% creating the networkx graph

import networkx as nx

# i create the networkx graph from the dataframe
G = nx.from_pandas_edgelist(df, 'user.id', 'retweeted_status.user.id',
                            create_using=nx.DiGraph(), edge_attr='weight')

#%% setting nodes attributes
# each node represents a user; i want to label it with the user.id and then
# i want to assign the user_annotation of the user as an attribute of the node


    #We make a dataframe to store the nodes
g_nodes = pd.DataFrame(G.nodes,columns=["user.id"])
    #We append to it the nodes which actually have a label
g_nodes = g_nodes.append(df, ignore_index=True)
    #We drop the duplicates, keeping the last duplicate (the one with an user_annotation)
g_nodes = g_nodes.drop_duplicates(subset=['user.id'],keep="last")
    #We fill the NaN values with "N/A" (leaving it blank as it is will produce errors later)
g_nodes = g_nodes.fillna("N/A")

    #We create a dictionary with the nodes (user.id) and attributes (user_annotation)
node_attr = g_nodes.set_index('user.id').to_dict('index') 
    #We add the attributes to the graph
nx.set_node_attributes(G, node_attr)
 



#%% let's see how the components of the graph look like

#this gives all the lengths (i.e. : number of nodes) of the components
components_sizes = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
# we have 65 components, but only the largest one is big (3705 nodes); the second
 # biggest is just 6 nodes and 22 isolated nodes! 

strong_components_sizes = [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
# we have 3369 (!) strongly connected components, the first one is 422 nodes, the second one just 5

#%% simple plot of the network with labels for the node and the weight, 
# just to check how it looks and if everything is ok
# taken from: https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx


pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)