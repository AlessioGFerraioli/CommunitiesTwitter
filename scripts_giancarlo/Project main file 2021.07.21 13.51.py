# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:31:44 2021

@author: gianc
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig 
import random


##############################################################################
#Preamble: Here we will use some variables in order to perform certain actions.

export_datafile_xlsx = False     #If True, the dataframe containing non-empty values of the column 'retweeted_status.user.id' will be exported to an Excel file.
export_weighted_xlsx = False     #If True, the dataframe containing the information about the users, the user they retweeted and how many times they did (the weight) will be exported to an Excel file.
export_annot_xlsx = False        #If True, the dataframe containing the information about how many times an user made a retweeted with the same polarity will be exported to an Excel file.

#%%
##############################################################################
#Reading the dataset

#We will use the following path to read the data
datafolder_path = r"D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj"
datafile = datafolder_path + r"\df_annotated_2021-01-26.pkl"

#In case we want to export the full data frame
datafile_csv = datafolder_path + r"\df_annotated_2021-01-26.csv"     #CSV
datafile_xlsx = datafolder_path + r"\df_annotated_2021-01-26.xlsx"   #Excel

#We read the datafile and make a dataframe
dataframe = pd.read_pickle(datafile)


##############################################################################
#Shrinking the data: only tweets that are retweets .

#We are interested in making a network where the nodes are users in Twitter and
#where the links (or edges) from user A to user B correspond to if the user A
#retweeted a tweet from user B.

#In this sense, we will keep only the rows where the column "retweeted_status.user.id"
#is not empty, meaning that the tweet in question is a retweet and the column
#will correspond to the id of the user of the original tweet.

#To to this, we do:
dataframe = dataframe[dataframe['retweeted_status.user.id'].notna()]


#If we want to export to a Excel (.xslx) file, we will get an error on the "created_at"
#column because of the format of the dates. Apparetly it doesn't amit that the date
#contains the timezone. In order to remove the timezone from the date we do:
    #Note: the timezone was originally UTC+2 and the information removed was "+02:00" for each date.

dataframe["created_at"] = dataframe["created_at"].dt.tz_localize(None)

#Now we export the dataframe to a Excel file

if export_datafile_xlsx :
    #Note: the dataframe is to big to fit into a single sheet on Excel, so we will
    #split the data to 500,000 rows per sheet.

    #We initialize the Xlsxwrite from pandas
    writer = pd.ExcelWriter(datafile_xlsx, engine='xlsxwriter')
    #We write on the .xlsx file
    dataframe.iloc[0:500000,:].to_excel(writer,sheet_name='Sheet1')
    dataframe.iloc[500000:1000000,:].to_excel(writer,sheet_name='Sheet2')
    dataframe.iloc[1000000:,:].to_excel(writer,sheet_name='Sheet3')
    #We save the file and close the writer
    writer.save()


##############################################################################
#Shrinking the data further: only tweets with information about the user's polarity towards the vaccine for COVID-19.

#We are interested in the position of the users of the tweets regarding the vaccine for COVID-19.
#This information is on the column "user_annotation" on the form of: AntiVax, Neutral or ProVax.
#We can perform further shrinking of the data on this regard.

dataframe_filtered = dataframe[dataframe['user_annotation'].notna()]


#%%
##############################################################################
#Analyzing the dataset: determining weights for the links

#Now we want to count the weight between each user interaction. 
#It's a measure of the "link" (edge) between them, which are related by a retweet.
    #The weight of an edge from user A to user B corresponds to how many times user A has retweeted
    #a tweet belonging to user B.


#For this, we will only care about the columns "user.id" and "retweeted_status.user.id".
dataframe_weight = dataframe_filtered[["user.id","retweeted_status.user.id"]]

#We count the number of times each user.id retweeted the same retweeted_status.user.id
    #We make a series to count the occurrences
weight_count = dataframe_weight.groupby(['user.id', 'retweeted_status.user.id']).size()
    #We add it to the dataframe
dataframe_weight = weight_count.to_frame(name = 'weight').reset_index()

#We can export it to a Excel file
if export_weighted_xlsx :
    weighted_xlsx = datafolder_path + r"\df_weightedRTs_2021-01-26.xlsx"
    dataframe_weight.to_excel(weighted_xlsx)


##############################################################################
#Analyzing the dataset: checking the polarity of each user

#We are interested in the polarity of each user.id, provided by the "user_annotation" column.
#We will perform a similar action as before to being able to notice two things:
    #1) How many times a single user.id has performed a Retweet.
    #2) Check if the same user.id has been labeled always with the same polarity (ProVax, Neutral or Antivax).

#We select the columns "user.id" and "user_annotation"
dataframe_annot = dataframe_filtered[["user.id","user_annotation"]]

#We count the number of times each user.id performed a retweet of the same polarity
    #We make a series to count the occurrences
annot_count = dataframe_annot.groupby(['user.id', 'user_annotation']).size()
    #We add it to the dataframe under the column "frequency"
dataframe_annot = annot_count.to_frame(name = 'frequency').reset_index()

#We can export it to a Excel file
if export_annot_xlsx :
    annot_xlsx = datafolder_path + r"\df_annotRTs_2021-01-26.xlsx"
    dataframe_annot.to_excel(annot_xlsx)

#Finally, we can check that there are no duplicates (meaning each user.id has been labeled according to the same polarity)

if len(dataframe_annot.index)==len(dataframe_annot.drop_duplicates().index):
    print("Each user.id has been labeled according to the same polarity.")
else:
    print("Some user.id has been labeled with more than one polarity.")


#%%
##############################################################################
#Reducing the noise within the data

#Since we are working with a big number of nodes and edges, it's possible to reduce
#the number of data as follows:
    #1) Consider only edges/links with a weight bigger than 1 (bigger or equal to 2)
    #2) Do not keep isolated nodes: it is, users than only performed retweets of their own tweets.

weight_cut = 2        #Only weights of this value or bigger will be considered.
isol_nodes = False    #If True, isol_nodes won't be filtered out.

#First, we filter according to the weight criteria:
dataframe_weighted = dataframe_weight[dataframe_weight["weight"]>=weight_cut]

#Lastly, we filter according to the isolated nodes criteria:
if not isol_nodes :
    #We will split the dataframe in 2: one including only self-retweets and one excluding them
        #The one with the self-retweets only
    df_selfRTs = dataframe_weighted[dataframe_weighted["user.id"]==dataframe_weighted["retweeted_status.user.id"]]
        #The one without the self-retweets
    df_noselfRTs = dataframe_weighted[dataframe_weighted["user.id"]!=dataframe_weighted["retweeted_status.user.id"]]
    #Now we check if among the self-retweets there are users that also made retweets from other users
    #To do this, we check if some "user.id" from the dataframe df_selfRTs are also present in df_noselfRTs
    selfRTs_check = df_selfRTs["user.id"].isin(df_noselfRTs["user.id"])
    #Now we keep only those user.id from df_selfRTs that are present in both
    df_selfRTs = df_selfRTs[selfRTs_check]
    #And we merge back the dataframes to a single one
    dataframe_weighted = pd.merge(df_selfRTs,df_noselfRTs, on=["user.id","retweeted_status.user.id","weight"],how='outer')


#%%
##############################################################################
#Now we proceed to created a graph to relate the users

#We create an empty directed graph
G = nx.DiGraph()

#We add the edges to the graph. Note: the nodes will be generated too
G.add_weighted_edges_from(dataframe_weighted.values, weight="weight") #Weights for the edges included



            
#%%
#Now we want to add attributes to the nodes.

#SUPER IMPORTANT NOTE
#WE HAVE 1986 DIFFERENT USERS BELONGING TO user.id WHO ACTUALLY PERFORMED A RETWEET.
#HOWEVER, WHEN IMPORTING ALL THE EDGES WE NOTE THAT THE NUMBER OF NODES INCREASES UP TO 10485.
#THIS IS BECAUSE WE HAVE SOURCE NODES (THE INITIAL 1986) AND TARGET NODES (THE REST).
#HERE WE HAVE TO CONSIDER THAT ONLY THE SOURCE NODES ARE LABELED BY POLARITY (user_annotation).

    #We make a dataframe to store the nodes
g_nodes = pd.DataFrame(G.nodes,columns=["user.id"])
    #We append to it the nodes which actually have a label
g_nodes = g_nodes.append(dataframe_annot, ignore_index=True)
    #We drop the duplicates, keeping the last duplicate (the one with an user_annotation)
g_nodes = g_nodes.drop_duplicates(subset=['user.id'],keep="last")
    #We fill the NaN values with "N/A" (leaving it blank as it is will produce errors later)
g_nodes = g_nodes.fillna("N/A")

    #We create a dictionary with the nodes (user.id) and attributes (user_annotation and frequency)
node_attr = g_nodes.set_index('user.id').to_dict('index') 
    #We add the attributes to the graph
nx.set_node_attributes(G, node_attr)



#%% i remove all the components of the graph with a number of nodes smaller than a threshold
# this is done to get rid of the people that only retweet/get retweeted by themselves or a 
# very small number of users, so they are not well integrated in the network
''' by alessio ''' 

for component in list(nx.weakly_connected_components(G)): 
    print(len(component))

delete_small_components = True

if delete_small_components == True:
    
    minimum_component_size = 7       # setting minimum_component_size we get rid of all the components except for the giant component
    
    for component in list(nx.weakly_connected_components(G)):
        if len(component) < minimum_component_size:
            for node in component:
                G.remove_node(node)

#%% visualizing 

#We create a list of colors to represent each polarity of the user_annotation. N/A values will be represented with the gray color.
color_map = []
for i in G.nodes:
    if G.nodes[i]['user_annotation']=="AntiVax":
        color_map.append('red')
    elif G.nodes[i]['user_annotation']=="Neutral":
        color_map.append('yellow')
    elif G.nodes[i]['user_annotation']=="ProVax":
        color_map.append('green')
    else:
        color_map.append('gray')


#We found the positions of the nodes to use for the graph
node_positions = nx.spring_layout(G,weight="weight") #this is a graph layout where if two nodes are connected by an edge, they attract each other, and if not, they repel each other

#We take the edge widths to use
edge_widths = list(nx.get_edge_attributes(G,'weight').values())  #widhts will be based on weights
edge_widths = np.fromiter(edge_widths, dtype=float)              #we convert it to a numpy array
edge_widths = np.log(edge_widths)                                #we take the log of the values, since some are too big compared to others


#We visualize the network
plt.figure()

#Now we created the layout for the graph

#First we plot the edges (so the nodes will appear on top and won't be covered by the edges)
nx.draw_networkx_edges(G,                   #the graph object we will use
        pos=node_positions,                 #position of the edges
        edgelist = G.edges(),               #edges
        width=edge_widths,                  #width of the edges
        edge_color = "royalblue",           #color of the links
        alpha=0.02,                          #transparency of the links
        arrowsize=5,                        #size of the arrows
        )

#Then we plot the nodes
nx.draw_networkx_nodes(G,                   #the graph object we will use
        pos=node_positions,                 #position of the nodes
        nodelist=G.nodes(),                 #nodes
        node_size=5,                        #size of the nodes
        node_color=color_map,               #color of the nodes
        alpha=0.5
        )

#save the plot
plt.savefig('D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/scripts_giancarlo/testimage.jpg')


#%%%% creating a igraph graph object from the networkx graph to do Communities detection with igraph
''' by alessio '''

# create undirected network from directed because most community detection algorithms use undirected
H = G.to_undirected(reciprocal=False, as_view=False)

# convert networkx graph to igraph graph
h = ig.Graph.from_networkx(H)

#%% communities detection

# finding the communities with the louvain algorithm
communities = h.community_multilevel(weights=dataframe_weight["weight"], return_levels=False)


print("number of communities found:")
print(len(communities))

#%%

def plot_communities(h, communities, file_name, vertex_size=3):
    
    ''' by alessio 
    
    h, igraph.Graph object
    communities, igraph.VertexClustering object
    file_name, str: the name of the file to which the plot is saved (include the directory)
    vertex_size, int (or even float?): size of vertices in the plot
    
    '''
    
    # SETTING NODES COLOR
    
    #creating a color map : a list of n random color codes, where n is the number of communities 
    cmap = ['#'+''.join([random.choice('0123456789abcdef') for x in range(6)]) for z in range(len(communities))]
    # saving the colors in a list; this list uses the same index of the nodes list,
    # so the j-th color corresponds to the j-th node. 
    #Different colors are assigned to member of different communities.
    vcolors = {v: cmap[i] for i, c in enumerate(communities) for v in c}
    # assigning colors from vcolors to nodes as an attribute
    h.vs["color"] = [vcolors[v] for v in h.vs.indices]
    
    # SETTING NODES SHAPE
    
    smap = ['rectangle', 'circle', 'triangle-up', 'diamond']
    vshapes = {v: smap[i] for i, c in enumerate(h.vs["user_annotation"]) for v in c}
    h.vs['shape'] = [vshapes[v] for v in h.vs.indices]
    
    
    
    # SETTING EDGES COLOR
    
    # saving the membership of each node (a list of which community each node belongs to)
    member = communities.membership
    # saving the colors in a list; this list uses the same index of the edges list,
    # so the j-th color corresponds to the j-th edge. 
    # all the edges within a community use the same colour. Edges between communities are grey 
    ecolors = {e.index: cmap[member[e.tuple[0]]] if member[e.tuple[0]]==member[e.tuple[1]] else "#e0e0e0" for e in h.es}
    # assigning colors from ecolors to edges as an attribute
    h.es["color"] = [ecolors[e] for e in h.es.indices]
   
    # SETTING EDGES WEIGHT  - wait, does this actually cancel the information on weight? should i use edge_width instead?
    
    # same as above but for weights. edges within a community have a much higher line weight
    eweights = {e.index: (3*h.vcount()) if member[e.tuple[0]]==member[e.tuple[1]] else 0.1 for e in h.es}
    # assigning colors from eweights to edges as an attribute
    h.es["weight"] = [eweights[e.index] for e in h.es]
    
    
    # the visual_style dict stores the attribute to pass to the plot function to control the visual appereance of the plot
    visual_style = {}
    visual_style["vertex_size"] = vertex_size
#    visual_style["vertex_shape"] = 
    visual_style["edge_color"] = '#00006001'
    visual_style["vertex_frame_width"] = 0
    visual_style["layout"] = h.layout_fruchterman_reingold(weights=h.es["weight"])
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    #plotting to file 
    ig.plot(h, file_name, mark_groups=True, **visual_style)
    
    return 



def find_higher_communities(communities):
    
    h_clusters = communities.cluster_graph(combine_edges='sum')
    macro_communities = h_clusters.community_multilevel(weights=h_clusters.es["weight"])

    return h_clusters, macro_communities


def find_macro_communities(communities, fixed_iter=None, max_iter=10, number_communities=None, prints=True):
    
    for i in range(max_iter):
         if (i == fixed_iter) or len(communities) == number_communities:
            i = max_iter + 1
            
         h_clusters, communities = find_higher_communities(communities)
       
    print("number of communities:")
    print(len(communities))
    return h_clusters, communities

            


# h_lvl3, communities_lvl3 = find_macro_communities(communities, fixed_iter=0)

plot_communities(h, communities,
      file_name="D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_giant_mark_groups.png")


#%% trying to see which user_annotation annotated user belongs to which community

''' by alessio 
2021.07.22 11.57
'''

#creating a list for the memberships for antivax users. in this list we will
#store the index of which community each antivax user belongs to. in this way
# we can count with maybe an histogram how many antivax users has each community.
#we do the same thing for provax, for neutral and for non-annotated users.
memberships_antivax = []
memberships_provax = []
memberships_neutral = []
memberships_non_annotated = []
for i in range(len(h.vs)):
    if h.vs['user_annotation'][i] == 'AntiVax':
        memberships_antivax.append(communities.membership[i])
    elif h.vs['user_annotation'][i] == 'ProVax':
        memberships_provax.append(communities.membership[i])
    elif h.vs['user_annotation'][i] == 'Neutral':
        memberships_neutral.append(communities.membership[i])
    else:
        memberships_non_annotated.append(communities.membership[i])
    

#%% histograms of memberships

''' by alessio
2021.07.22 11.57
'''

hist = False

if hist == True:
    fig, ax = plt.subplots()
    bins = np.linspace(0, len(communities)-1, len(communities))
    alpha = 0.3
    ax.hist(memberships_antivax, bins, alpha=alpha, label='AntiVax')
    ax.hist(memberships_provax, bins, alpha=alpha, label='ProVax')
    ax.hist(memberships_neutral, bins, alpha=alpha, label='Neutral')
    ax.hist(memberships_non_annotated, bins, alpha=alpha, label='No data')
    ax.legend(loc='upper right')
    ax.set_xlabel('Communities')
    ax.set_ylabel('Users count')
    
    plt.show()
    
