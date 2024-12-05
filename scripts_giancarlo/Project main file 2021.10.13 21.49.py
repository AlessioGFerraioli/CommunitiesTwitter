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
#Reading the dataset - it takes some time

#We will use the following path to read the data
datafolder_path = r""
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
# also most centrality measures work only on connected networks
''' by alessio ''' 

# for component in list(nx.weakly_connected_components(G)): 
#     print(len(component))

delete_small_components = True

if delete_small_components == True:
    
    minimum_component_size = 7       # setting minimum_component_size we get rid of all the components except for the giant component
    
    for component in list(nx.weakly_connected_components(G)):
        if len(component) < minimum_component_size:
            for node in component:
                G.remove_node(node)

#%% visualizing  WARNING: TAKES A LOT

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
'''

by alessio 

'''

# finding the communities with the louvain algorithm
communities = h.community_multilevel(weights=dataframe_weight["weight"], return_levels=False)


print("number of communities found:")
print(len(communities))

#%% visualizing the communities
''' by alessio '''


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

            

plot_communities(h, communities,
      file_name="D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_testtt.png")


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
    

#%% histograms of memberships to communities

''' by alessio
2021.07.22 11.57
'''

hist = True

if hist == True:
    fig, ax = plt.subplots()
    bins = np.linspace(0, len(communities)-1, len(communities))
    alpha = 0.3
    ax.hist(memberships_antivax, bins, alpha=alpha, label='AntiVax')
    ax.hist(memberships_provax, bins, alpha=alpha, label='ProVax')
    ax.hist(memberships_neutral, bins, alpha=alpha, label='Neutral')
    #ax.hist(memberships_non_annotated, bins, alpha=alpha, label='No data')
    ax.legend(loc='upper right')
    ax.set_xlabel('Communities')
    ax.set_ylabel('Users count')
    
    plt.show()
    
#%%
#%%#
#%%
#%%
#%%
#%%
#%%
#%%

#%% ANALYZING THE DEGREE DISTRIBUTION of the (undirected) largest component

''' BY ALESSIO, 2021.10.05

This section is preparatory to all the sections that involve centrality measures
because here I write some functions that I will use to analyze and plot the 
centrality measures. I also evaluate here the degrees centrality - which is
the simplest centrality measure - on the whole graph just to have something
to work on
WE CAN FIND HERE:
    
    -HISTOGRAM OF DEGREE CENTRALITIES
    -HISTOGRAM OF DEGREE CENTRALITIES WITH LOGARITHMIC BINNING AND FITTING OF POWER LAW
    -CUMULATIVE DEGREE DISTRIBUTION (it is apparently correct but i don't 100% trust the plot)
    - (failed) ATTEMP AT CUMULATIVE DEGREE DISTRIBUTION (same as before, i used a different reasoning but the function does not work and now i can't seem to find why)


'''




def find_alpha(degree_centr, min_deg):

    ''' this function returns alpha (i.e.: the exponent of the power law)
    found by analytical methods, see pag 258 of Newman, Network an Introduction
    
    inputs:  degreee_centr, the array containing the degree centralities
             min_deg, int, the minimum degree from which we start evaluating alpha
             (note that the power law does not hold for very small degrees)
             
    output:  alpha, float, exponent of the power law
    
    '''    

    N = len(degree_centr)      #total number of vertices
    
    degree_centr.sort()
    
    j = 0
    while degree_centr[j] < min_deg:
        j = j+1

    n = N - (j+1)
    summation = 0   # this variable is the summation in (8.6), pag 258, Newmann
    check = 0
    for i in range(j, N-1):
        check = check +1
        summation = summation + np.log(degree_centr[i]/(min_deg-0.5))
    
    alpha = 1 + n/summation
    
    if check != n:
        print("wutathefakuru")
    return alpha



def log_bin_hist(data, exp_start=None, exp_stop=None, number_of_bins=None, 
                 bin_factor=2):
    
    """
    
    This function outputs returns the bins, widths and hist_norm defining
    an histogram with logarithmic binning of data. 
    Logarithmic binning means that the size of each bin is equal to the size
    of the previous bin times a constant factor named bin_factor.
    The bins are normalized to account for the fact that bins have different sizes.
    
    
    inputs:       data,       
                              an 1D-array of data to plot
                  exp_start,  
                              the exponent from which the binning would start.
                              default is None, which uses the log of the minimum of data
                  exp_stop,  
                              the exponent from which the binning would stop.
                              default is None, which uses the log of the maximum of data
                  number_of_bins,
                              the number of bins created. 
                              default is None, which uses the number of bins that 
                              arises by using a constant multiplicative factor for the size
                              equal to bin_factor
                  bin_factor, 
                              the numerical factor by which the width of each bin
                              is multiplicated to get the width of the next bin.
                              default is 2 because it is the most used and useful value
                              as stated in newman, network an introduction pag 254
                              
    outputs:     bins, widths, hist_norm
    
    requires :   numpy as np
                 matplotlib.pyplot as plt
                            
                 
    adapted from : https://stackoverflow.com/questions/37170511/scaled-logarithmic-binning-in-python

    """
    
        
    if exp_start is None:
        exp_start = np.log(min(data))
        
    if exp_stop is None:
        exp_stop = np.log(max(data))    
        
    if number_of_bins is None:
        number_of_bins = round((exp_stop-exp_start)/np.log(bin_factor)) 
        # round function is used to round up the float to the nearest integer
        
    
    # log-scaled bins
    bins = np.logspace(exp_start, exp_stop, number_of_bins)
    widths = (bins[1:] - bins[:-1])
    
    # Calculate histogram
    hist = np.histogram(data, bins=bins)
    # normalize by bin width
    hist_norm = hist[0]/widths
    
   
    return bins, widths, hist_norm


def plot_log_bin_hist(data, exp_start=None, exp_stop=None, number_of_bins=None, 
                 bin_factor=2, fit=False, degree_min=5, 
                 use_analytical_alpha=True):
    
    """
    
    This function plots with matplotlib an histogram on a log-log scale with
    logarithmic binning, meaning that the size of each bin is equal to the size
    of the previous bin times a constant factor named bin_factor. The histogram
    is evaluated with the log_bin_hist function.
    The bins are normalized to account for the fact that bins have different sizes.
    If fit is True, it fits the histogram to a power law with igraph and 
    displays the fit on the histogram.
    
    
    N.B.: there's a little problem with exp_start and exp_stop. Inside this
    function they are passed to the log_bin_hist function, but if they are None
    there is an error when passing them. I don't know why and I don't care 
    because solving this problem is a waste of time. Just pass exp_start 
    and exp_stop as integers and you will be fine. (0,2) or (0,3) works best.
    
    
    inputs:       data,       
                              an 1D-array of data to plot
                  exp_start,  
                              the exponent from which the binning would start.
                              default is None, which uses the log of the minimum of data
                  exp_stop,  
                              the exponent from which the binning would stop.
                              default is None, which uses the log of the maximum of data
                  number_of_bins,
                              the number of bins created. 
                              default is None, which uses the number of bins that 
                              arises by using a constant multiplicative factor for the size
                              equal to bin_factor
                  bin_factor, 
                              the numerical factor by which the width of each bin
                              is multiplicated to get the width of the next bin.
                              default is 2 because it is the most used and useful value
                              as stated in newman, network an introduction pag 254
                  fit,       
                              bool, if True it will fit the power law with igraph
                              and display
                  degree_min, 
                              float. the minimum degree value from which the fit should start.
                              n.b. the power law does not hold for small degree, so the
                              smallest values should be ignored for the fit. Default is 5. 
                  use_analytical_alpha,
                              float. If it is True, alpha (the exponent of the power law distr)
                              will be calculated in an analytic way (see pag. 258 of Newmann).
                              If it is False, it will be evaluated from standard fitting methods
                              on the histogram. This is not ideal because the assumption of 
                              indipendence of the bins are not met here. 
                              The multiplicative parameter of the power law (i.e. normalization)
                              is in any case evaluated from the fitting of the histogram -
                              I know, i know..
                              Default is True.
                              
    outputs:     alpha
    
    requires :   numpy as np
                 matplotlib.pyplot as plt
                 igraph as ig
                 find_alpha
                 log_bin_hist
                 
    """
    
    #find alpha the power law exponent
    alpha = find_alpha(data, degree_min)  
    
    #i evaluate the histogram
    bins, widths, hist_norm = log_bin_hist(data, exp_start, exp_stop,
                                           number_of_bins, bin_factor)
    
    # plot it
    fig, ax = plt.subplots(1, 1)    
    ax.bar(bins[:-1], hist_norm, widths)
    if fit == True:   # fitting and plotting the fit
        fit_params = ig.power_law_fit(degree_centr, xmin=degree_min, return_alpha_only=False)
        
        if use_analytical_alpha == False:
            alpha = fit_params.alpha
        elif use_analytical_alpha == True:
            alpha = find_alpha(data, degree_min)
        x = np.linspace(10**exp_start,10**exp_stop, 1000)
        ax.plot(x, -fit_params.L*x**(-alpha), color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    return alpha
    





def plot_cumulative_degree(degree_centr, absolute_number=False,
                                plot=True, fit=False, degree_min=15):
    
    """
    This function plots with matplotlib the cumulative distribution of degrees
    on a log-log scale; the cumulative distribution of degrees is the function
    that links to a specific degree value the fraction of vertices that has a
    degree value equal or greater than that. 
    
    see pag 257 of newman, networks an introduction.
    
    
    inputs:       degree_centr, an 1D-array containing the degrees of a network
                  absolute_number, bool. If True the function returns the absolute
                               number of vertices for each point instead of the fraction. 
                               Default is False.
                  plot, bool. If True the function will plot the cumulative distribution
                              of degree in log-log scale. Default is True.
    outputs:     cumulative_degree, array
    requires :   numpy as np
                 matplotlib.pyplot as plt
                 
    """
 
    n = len(degree_centr)
    
    degree_centr.sort()
       
    cumulative_degree = np.zeros(max(degree_centr)+1)  # the +1 it's because it counts from 0
    
    for i in range(len(cumulative_degree)):
        for j in range(len(degree_centr)):
            if i == degree_centr[j]:
                cumulative_degree[i] = len(degree_centr)-j  #the number of elements with that number of degree or greater is all the element after j and j itself. nb len gives the len, but the index arrives to len-1. so it's really (len()-1) - (j-1)
                if absolute_number == False:    # normalizing over number of vertices
                    cumulative_degree[i] = cumulative_degree[i]/n
                break

    #removing the zeros created by this process
    for k in range(len(cumulative_degree)-1):
        if cumulative_degree[k]==0:
            i = k 
            while cumulative_degree[k]==0 and i<(len(cumulative_degree)-1):
                cumulative_degree[k] = cumulative_degree[i+1]
                i = i+1
        

    # plot it
    fig, ax = plt.subplots(1, 1)    
    ax.plot(cumulative_degree)
    if fit == True:   # fitting and plotting the fit
        fit_params = ig.power_law_fit(degree_centr, degree_min, return_alpha_only=False)
        x = np.linspace(0,max(degree_centr)+1, 1000)
        if absolute_number==False:
            ax.plot(x, -fit_params.L*x**(-fit_params.alpha)/n, color='red')
        else:
            ax.plot(x, -fit_params.L*x**(-fit_params.alpha), color='red')

   
    ax.set_xscale('log')
    ax.set_yscale('log')
   
    return cumulative_degree





#this next function does not work, i don't know why. i suspect it's
# a trivial problem but whatever
def evaluate_cumulative_degree3(degree_centr, absolute_number=False,
                                plot=True, fit=False):
    
    """
    
    IT DOES NOT WORK - I DON'T UNDERSTAND WHY 
    
    This function plots with matplotlib the cumulative distribution of degrees
    on a log-log scale; the cumulative distribution of degrees is the function
    that links to a specific degree value the fraction of vertices that has a
    degree value equal or greater than that. 
    
    see pag 257 of newman, networks an introduction.
    
    
    inputs:       degree_centr, an 1D-array containing the degrees of a network
                  absolute_number, bool. If True the function returns the absolute
                               number of vertices for each point instead of the fraction. 
                               Default is False.
                  plot, bool. If True the function will plot the cumulative distribution
                              of degree in log-log scale. Default is True.
    outputs:     cumulative_degree, array
    requires :   numpy as np
                 matplotlib.pyplot as plt
                 
    """
 
    n = len(degree_centr)
    
    degree_centr.sort()
       
    cumulative_degree = np.zeros(max(degree_centr)+1)  # the +1 it's because it counts from 0

    
    j = 0
    for i in range(n-1):
        print(i)
        while degree_centr[j] <= 1+i: 
            cumulative_degree[1+i] = n-j
            i = i+1
            if i+1 >= n:
                break
            
            
        print(i)
        
        j = j+1
        
        if j == len(cumulative_degree):
            break
        else:
            
            while degree_centr[j] == degree_centr[j-1]:
                j = j+1

    # plot it
    fig, ax = plt.subplots(1, 1)    
    ax.plot(cumulative_degree)
    if fit == True:   # fitting and plotting the fit
        fit_params = ig.power_law_fit(degree_centr, xmin=5, return_alpha_only=False)
        x = np.linspace(0,max(degree_centr)+1, 1000)
        if absolute_number==False:
            ax.plot(x, -fit_params.L*x**(-fit_params.alpha)/n, color='red')
        else:
            ax.plot(x, -fit_params.L*x**(-fit_params.alpha), color='red')

   
    ax.set_xscale('log')
    ax.set_yscale('log')
   
    return cumulative_degree


# --------------

#%%% CENTRALITY MEASURES ON THE WHOLE GRAPH

''' 

CENTRALITY MEASURES ON THE WHOLE GRAPH

by alessio , 2021.10.05 


in this section i will evaluate and eventually plot the different centrality
measures of our whole graph. 
the idea is to do in the next section the same measures for the communities 
inside the graph and compare their behaviour among them and with the whole graph

i'm doing these measures on a graph that's only the largest component of the
complete graph. i will also consider the graph to be undirected. 

maybe i will do the measures considering the graph as directed 

'''



# DEGREE
#the simplest property one can think of is the vertex degree. 
#The degree of a vertex equals the number of edges adjacent to it. '''

#this returns a list of the degree for each node
degree_centr = h.degree()
#plotting histogram of degree in a log scale
plt.hist(np.log(degree_centr), bins=15)

# NOTE THAT it's a right skewed distribution. this is to
# be expected in most real networks, meaning that there are a small number
# of vertices that have a very high degree, while most of the vertices
# have a small degree

# I ask the question: IS THIS NETWORK A SCALE-FREE NETWORK? (a network that follows the power law)
# how to do this? see pag 254

# # from pag 248:
#     In Section 8.4 we discuss so-called
# “scale-free” networks, i.e., networks with power-law degree distributions. Such networks are
# believed to have an unusual structure consisting of a central “core” to the network that contains
# most of the vertices and has a mean geodesic distance between vertex pairs that scales only as log
# log n with network size, and not as log n, making the mean distance for the whole network scale as
# log log n also. Outside of this core there are longer “streamers” or “tendrils” of vertices attached to
# the core like hair, which have length typically of order log n, making the diameter of the network
# # of order log n
# This sort of behavior could be detected by measuring separately the mean
# geodesic distance and diameter of networks of various sizes to confirm that they vary differently
# with n. (It’s worth noting, however, that behavior of the form log log n is very difficult to confirm
# in real-world data because log log n is a very slowly varying function of n.)


# i can do the following:

# plotting the logarithmic binned histogram and evaluating the alpha 
# (coefficient of the power law) with the analytical method (if use_analytical_alpha==True)
# or by fitting of the histogram (if use_analytical_alpha==False). 
# N.B.: the analytical method is the one you should use because the fitting 
# isn't proper since the bins are not independet from one another
alpha = plot_log_bin_hist(degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False)
# from this i see that the log bin histogram follows a straight line, so
# this confirms the fact that this network is a scale free one
    
# i could just evaluate alpha in the analytical way without plotting anything:
alpha = find_alpha(degree_centr, min_deg=15)
    
# i could also evaulaute the cumulative degree distribution and plotting it by using
# # the plot_cumulative_degree() function. this should be more or less equivalent
# to the histogram, in the sense that it gives us the same information. for some
# reason i have problem with the fitted coefficient of the power law, because the
# line that represents the fit has the right angular coefficient but it is not
# aligned with the cumulative degree curve. whatever, i will simply not use the fit
# here
cumulative_degree = plot_cumulative_degree(degree_centr,
                                                absolute_number=False,
                                                fit=False)



# ----------

# EIGENVECTOR CENTRALITY 
# for eigenvector centrality, the importance of a vertex is not measured only
# with the number of vertices it's connected to, but also taking into account
# if it's connected to important vertices
# see more at pag 171 of the book and at:
# https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#eigenvector_centrality

#this returns a list of the eigenvector centrality for each node
eigen_centr = h.eigenvector_centrality(directed=True, scale=True, weights=None, return_eigenvalue=False)
#plotting histogram of eigenvector centralities in a log scale
plt.hist(np.log(eigen_centr), bins=20)


# --------------


# CLOSENESS CENTRALITY
# it's the inverse of the mean distance of a vertex to the other vertices; 
# vertices with a small mean distance have high closeness centrality
# we expect that the values do not change much, usually there is a factor of
# 5 or less between the highest and lowest value
# see more at pag 184 of the book and at:
# https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#closeness

#this returns a list of the closeness centrality for each node
closeness_centr = h.closeness(weights=None)
#plotting histogram of closeness centralities in a log scale . no non fare log scale perché spannano poco
plt.hist((closeness_centr), bins=20)

max(closeness_centr)
min(closeness_centr)
#note that the difference between max and min is pretty small

# SMALL-WORLD EFFECT
# An interesting property of entire networks, which is related to the closeness centrality, is the
# mean geodesic distance between vertices. In Section 8.2 we will use measurements of mean
# distance in networks to study the so-called “small-world effect.”

# i want to evaluate the mean geodesic distance to show the small world effect.
mean_geodesic_path = h.average_path_length(directed=False, unconn=True)
# it's less than 4!!! god that's small!!


# --------------


# HARMONIC CENTRALITY
# a modification of the closeness centrality in which instead of measuring
# the mean distance we measure the harmonic mean distance, i.e. the mean of 
# the inverse distances. from the book:
# "This definition has a couple of nice properties. First, if dij = inf,
# because i and j are in different  components, then the corresponding term
# in the sum is simply zero and drops out. Second, the
# measure naturally gives more weight to vertices that are close to i than to those far away.
# Intuitively we might imagine that the distance to close vertices is what matters in most practical
# situations—once a vertex is far away in a network it matters less exactly how far away it is, and
# the harmonic centrality reflects this, having contributions close to zero from all such vertices.
# Despite its desirable qualities, however, harmonic centrality is rarely used in practice. We have seen it
# employed only occasionally."
# https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#harmonic_centrality

#this returns a list of the harmonic centrality for each node
harm_centr = h.harmonic_centrality(weights=None)
#plotting histogram of harmonic centralities in a log scale
plt.hist(np.log(harm_centr), bins=20)


# SEE ALSO : HUBS AND AUTHORITIES, pag 180
#In the case of directed networks, there is another twist to the centrality measures introduced in this
#section


# -------------

# BETWEENNESS CENTRALITY
# it's the number of shortest paths that pass through a vertex
# note that a vertex can have very low degree/eigenvector centr or closeness 
# and still have high betweeness 
# values of betweenness are expected to be over a wide range, they can span up 
# to ~n/2  (n number of nodes); real values usually do not span so much
# https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#betweenness

#this returns a list of the betweenness centrality for each node
between_centr = h.betweenness(vertices=None, directed=True, cutoff=None, weights=None)
#plotting histogram of betweenness centralities in a log scale
plt.hist(between_centr)
max(between_centr)
min(between_centr)
#note that the difference between max and min is very large

# ------------

#CLUSTERING COEFFICIENTS ???  (another possible centrality measure)

# --------------


#SALIENT LINKS???  (another possible centrality measure)



#%% COMPARING THE CENTRALITIES CORRELATIONS FOR THE WHOLE GRAPH
'''

COMPARING THE CENTRALITIES CORRELATIONS FOR THE WHOLE GRAPH

by alessio,  2021.10.05

in this section i will plot one centrality against the other to see their 
correlation and which behaviours may arise.

'''


s = 0.4    # this is the size of the points of the scattergraph

plt.scatter(closeness_centr, degree_centr, s) #it's almost a gaussian lol
plt.scatter(harm_centr, degree_centr, s) # same as above but a little bit shifted to the left
plt.scatter(eigen_centr, degree_centr, s) 
plt.scatter(closeness_centr, eigen_centr, s) 
plt.scatter(harm_centr, eigen_centr, s) 
plt.scatter(closeness_centr, harm_centr, s)  #they are mostly the same, as i expect
plt.scatter(between_centr, degree_centr, s) 
plt.scatter(between_centr, closeness_centr, s) 
plt.scatter(between_centr, harm_centr, s) 
plt.scatter(between_centr, eigen_centr, s) 






#%% CENTRALITY MEASURES ON COMMUNITIES

'''

CENTRALITY MEASURES ON COMMUNITIES

 by Alessio , 2021.10.11



in this section i measure centrality measures for the different communities
found in the community detection section.
I compare those measures by plotting them as histograms with logarithm binning.
n.b in order to compare should i normalize by the number of vertices in the community?
yes. but it is already done by default by the igraph functions when computing
the centrality measures
 
'''



# i create arrays in which i will store the centrality measures for the different communities
number_of_communities = len(communities)
degrees_coms = [ [] for _ in range(number_of_communities) ]
between_coms = [ [] for _ in range(number_of_communities) ]
closeness_coms = [ [] for _ in range(number_of_communities) ]
harm_coms = [ [] for _ in range(number_of_communities) ]
eigen_coms = [ [] for _ in range(number_of_communities) ]


# I need a way to access each community as a subgraph
## it's as simple as that: if you have a ig.clustering.VertexClustering object
# (like the output of community detection), to access a subgraph you just
# use the method subgraph.
# now i can evaluate centrality measures for each community and store them in
# the arrays i've just created
for i in range(number_of_communities):
    print(i)
    degrees_coms[i] = communities.subgraph(i).degree()
    between_coms = communities.subgraph(i).betweenness()
    closeness_coms = communities.subgraph(i).closeness()
    harm_coms = communities.subgraph(i).harmonic_centrality()
    eigen_coms = communities.subgraph(i).eigenvector_centrality()





#I should plot those histograms on the same graph. 
def compare_hist(data, opacity=0.3, number_of_bins=20, legend=None, title=None, 
                 x_scale_log=False, y_scale_log=False):
    
    ''' this function plots 2 or more histograms on the same
    figure. returns nothing.
    '''
        
    fig, ax = plt.subplots()
    
    for i in data:
        
        ax.hist(i, bins=number_of_bins, alpha=opacity)

    if legend is not None:
         ax.legend(legend)
        
    if title is not None:
         ax.set_title(title)
         
    if x_scale_log == True:
          ax.set_xscale('log')
          
    if y_scale_log == True:      
          ax.set_yscale('log')
   
        

# and also if i want to compare log binned histograms:            
def compare_log_bin_hist(data, opacity=0.3, exp_start=0,
                         exp_stop=3, number_of_bins=20,
                         legend=None, title=None):
    
    ''' this function plots 2 or more histograms in log binning on the same
    figure. returns nothing.
    '''
        
    fig, ax = plt.subplots()
    
    for i in data:
        bins, widths, hist_norm = log_bin_hist(i, exp_start, exp_stop,
                                           number_of_bins)
        ax.bar(bins[:-1], hist_norm, widths, alpha=opacity)
            

    ax.set_xscale('log')
    ax.set_yscale('log')
        
    
    if legend is not None:
         ax.legend(legend)
         
    if title is not None:
         ax.set_title(title)
         
        


# for example, i plot on the same histogram the degrees of communities 7, 8 ,9
compare_log_bin_hist(degrees_coms[6:9],
                     legend=("Community 7", "Community 8","Community 9"),
                     title="Degrees in log binning for communities 7, 8 and 9")
                     


#%%% CENTRALITY MEASURES FOR ANTIVAX,NEUTRAL,PROVAX

'''

CENTRALITY MEASURES FOR ANTIVAX,NEUTRAL,PROVAX

 by alessio,  2021.10.13
 

in this section i want to evaluate different centrality measures on the
three groups: antivax, novax, neutrals and compare them by plotting 
them as logarithmic binned histograms.

 
'''



#I can use the ig.Graph.VertexSeq.select('condition')  function to access
# only a subset of the ig graph that satisfies some condition

# compare the degrees for the three categories
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').degree(),
                      h.vs.select(user_annotation='ProVax').degree(),
                      h.vs.select(user_annotation='Neutral').degree()),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Degrees in logarithmic binning')  

# compare the betweenness for the three categories with log binning
# n.b. the calculation of betweenness is the slowest
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').betweenness(),
                      h.vs.select(user_annotation='ProVax').betweenness(),
                      h.vs.select(user_annotation='Neutral').betweenness()),
                    legend=('AntiVax', 'ProVax', 'Neutral'),
                    title='Betwenness centrality in logarithmic binning')     

# compare the betweenness for the three categories with log y axis
# n.b. the calculation of betweenness is the slowest
compare_hist((h.vs.select(user_annotation='AntiVax').betweenness(),
                      h.vs.select(user_annotation='ProVax').betweenness(),
                      h.vs.select(user_annotation='Neutral').betweenness()),
                    legend=('AntiVax', 'ProVax', 'Neutral'),
                    title='Betwenness centrality with log y axis',
                    x_scale_log=False, y_scale_log=True)     


# compare the closeness for the three categories with normal histogram
compare_hist((h.vs.select(user_annotation='AntiVax').closeness(),
                      h.vs.select(user_annotation='ProVax').closeness(),
                      h.vs.select(user_annotation='Neutral').closeness()),
             legend=('AntiVax', 'ProVax', 'Neutral'),
             title='Closeness centrality')   
 
# compare the harmonic centrality for the three categories with normal histogram
#this does not work because harmonic centrality is not recognized wtf??
# well we can live without it since it is just a more sophisticated closeness..
compare_hist((h.vs.select(user_annotation='AntiVax').harmonic_centrality(),
                      h.vs.select(user_annotation='ProVax').harmonic_centrality(),
                      h.vs.select(user_annotation='Neutral').harmonic_centrality()),
             legend=('AntiVax', 'ProVax', 'Neutral'),
             title='Harmonic centrality')   
 

# compare the eigenvector centrality for the three categories
#this does not work because eigenvector centrality is not recognized wtf??
# well we can live without it since it is just a more sophisticated degree centrality..
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').eigenvector_centrality(),
                      h.vs.select(user_annotation='ProVax').eigenvector_centrality(),
                      h.vs.select(user_annotation='Neutral').eigenvector_centrality()),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Eigenvector centrality')    





#%%%  ASSORTATIVITY COEFFICIENT

'''
by alessio , 2021.10.18

ASSORTATIVITY COEFFICIENT

the assortativity coefficient gives a measure of how different types of nodes mix.
The assortativity coefficient is one if all the connections stay within 
categories and minus one if all the connections join vertices of different categories.

n.b. the community detection algorithms work by cutting the edges for which
we would have maximum assortativity coefficient based on some metric (for example betwenness)
'''

# i use networkx because i can't get to work the corresponding igraph function, 
# while this is very easy to use
assortativity_coefficient = nx.attribute_assortativity_coefficient(G, "user_annotation")
# the assirtativity coefficient is 0.17418246698301018, which is positive =>
#  people tend to connect with people of the same group 



# %%%   stuff

#this gives the "user_annotation" of the vertex of h with the highest degree
h.vs.select(_degree=h.maxdegree())["user_annotation"]
# we could use it to see who are the most connected users,
# or expanding on that (including other data from the dataframe)
# we could see who are the users who post more, who post more links etc.-

#----
 
# from the igraph tutorial:
    
#     _between takes a tuple consisting of two VertexSeq objects or lists containing vertex indices or Vertex objects and selects all the edges that originate in one of the sets and terminate in the other. E.g., to select all the edges that connect men to women:

# men = g.vs.select(gender="m")
# women = g.vs.select(gender="f")
# g.es.select(_between=(men, women))
# i could use this to see the edges from a antivax to a provax etc


