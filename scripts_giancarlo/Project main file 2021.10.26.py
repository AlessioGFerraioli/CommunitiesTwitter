# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 23:14:32 2021

@author: Utente
"""


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig 
import random
import leidenalg as la

preprocess = False
remove_tweets_with_non_annotated_user = True
weight_cut = 2   # during preprocessing, remove all edges w weight lower thatn weight cut
SAVE_DF = False   # True to save the preprocessed dataframe in a file
READ_DF = True   # true to load the preprocessed datafram from file

visualize_network = False

LOUVAIN = False  # if true, the community detection will be done with the louvain algorithm instead of leiden (default)
DIRECTED_COMMUNITY_DETECTION = True 
WEIGHTED_COMMUNITY_DETECTION = True
AVERAGE_ALGORITHM = False

visualize_communities = False
visualize_communities_2 = False


#%% Reading the dataset - it takes some time

if preprocess == True:
    #We will use the following path to read the data
    datafolder_path = r""
    datafile = datafolder_path + r"\df_annotated_2021-01-26.pkl"
    
    #We read the datafile and make a dataframe
    df = pd.read_pickle(datafile)

#%% preprocessing the dataframe: 
    # - remove tweets for which we don't have a user_annotation
    # - remove tweets that are not retweets
    # - count how many times each user A retweeted each user B 
    # - remove all the weights under 2
    # - save the processed dataframe to file
    
if preprocess == True:
   
    if remove_tweets_with_non_annotated_user == True:
        # i remove from my dataframe the rows for which user_annotation is nan
        # (we only want to keep the user with a user_annotation)
        df = df.loc[~df["user_annotation"].isna()]
        
    # i remove from my dataframe the rows for which "retweeted_status.id" is nan
    # (we only want to keep the tweets that are retweets)
    df = df.loc[~df["retweeted_status.id"].isna()]
    
    # we count how many times user A retweeted user B and store the values in a "weight" attribute
    df['weight'] = df.groupby(['user.id', 'retweeted_status.user.id'])['user.id'].transform('size')
    
    
    #i remove all the edges with weight<weight_cut
    df = df.loc[df["weight"] >= weight_cut]
    
    if SAVE_DF == True:
        #saving to a file to not do this every time             
        df.to_csv(r"D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj\df_processed_wc2.csv")

#%% creating the networkx graph

if READ_DF == True:
    #reading the processed dataframe from file
    df = pd.read_csv(r"D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj\df_processed_wc2.csv")            


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


#%% i want to keep only the largest component

# now my graph G that I've saved is just the largest component
G = G.subgraph(max(nx.weakly_connected_components(G), key=len))

#%% i create lists of nodes for antivaxxers, provaxxers, neutrals
# what's the purpose of these lists?


antivaxxers = []
provaxxers = []
neutrals = []
non_annotated = []

for (p, d) in G.nodes(data="user_annotation"):
    if d == "AntiVax":
        antivaxxers.append(p)
    elif d == "ProVax":         
        provaxxers.append(p)
    elif d == "Neutral":         
        neutrals.append(p)
    else:         
        non_annotated.append(p)
        


#%% i create a igraph graph object from the networkx graph to do Communities detection and stuff with igraph
''' by alessio '''


if DIRECTED_COMMUNITY_DETECTION == False:
    # i create an undirected network from a directed one because most community 
    # detection algorithms require undirected networks
    H = G.to_undirected(reciprocal=False, as_view=False)
    
    # convert networkx graph to igraph graph
    h = ig.Graph.from_networkx(H)
else:
    # convert directed networkx graph to igraph graph
    h = ig.Graph.from_networkx(G)

#%% USEFUL FUNCTIONS FOR THE CENTRALITY MEASURES

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
    - COMPARE HISTOGRAMS (to plot 2 or more histogram on the same plot)                                                     
    - COMPARE LOG BIN HISTOGRAMS (to plot 2 or more histogram in log binning on the same plot)                                                     



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
    
    This function returns the bins, widths and hist_norm defining
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
                 use_analytical_alpha=True, title=None, xlabel=None, ylabel=None):
    
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
        fit_params = ig.power_law_fit(data, xmin=degree_min, 
                                      return_alpha_only=False)
        
        if use_analytical_alpha == False:
            alpha = fit_params.alpha
        elif use_analytical_alpha == True:
            alpha = find_alpha(data, degree_min)
        x = np.linspace(10**exp_start,10**exp_stop, 1000)
        ax.plot(x, -fit_params.L*x**(-alpha), color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if title is not None: 
        ax.set_title(title)
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)  
    
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
    ax.set_xlabel('Degree k')
    ax.set_ylabel('Fraction of nodes having degree k or higher')
    
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




#I should plot those histograms on the same graph. 
def compare_hist(data, opacity=0.3, number_of_bins=20, legend=None, title=None, 
                 x_scale_log=False, y_scale_log=False, x_label=None, y_label=None):
    
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
   
       
    if x_label is not None:    
        ax.set_xlabel(x_label)
        
    if x_label is not None:    
        ax.set_ylabel(y_label)
        

# and also if i want to compare log binned histograms:            
def compare_log_bin_hist(data, opacity=0.3, exp_start=0,
                         exp_stop=3, number_of_bins=20,
                         legend=None, title=None, xlabel=None, ylabel=None):
    
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
         
    if xlabel is not None:
         ax.set_xlabel(xlabel)        
        
    if ylabel is not None:
         ax.set_ylabel(ylabel)    

# --------------

#%%% CENTRALITY MEASURES ON THE WHOLE GRAPH AS AN UNDIRECTED UNWEIGHTED NETWORK WITH IGRAPH
#  N.B. : further down in the program you will find the same measures done with
# networkx, considering the graph as directed and sometimes taking into account also
# the weights. i don't know if these measures could tell us something that
# the ones done on the directed network won't.

''' 

CENTRALITY MEASURES ON THE WHOLE GRAPH

WITH IGRAPH


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
fig, ax = plt.subplots()
ax.hist(np.log(degree_centr), bins=15)
ax.set_title('Degree centrality histogram')
ax.set_ylabel('Number of nodes having degree k')
ax.set_xlabel('Degree k in log scale')


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
                          fit=True, use_analytical_alpha=False,
                          title="Degree centrality histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

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
plt.title('Cumulative degree centrality')



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
plt.title('eigenvalue centrality histogram')


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
plt.title('Closeness centrality histogram on the whole network')
plt.set_ylabel('Number of nodes having closeness C')
plt.set_xlabel('Closeness centrality C')

   
# plot it
fig, ax = plt.subplots(1, 1)    
ax.hist((closeness_centr), bins=20)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_title('Betweenness centrality histogram')
ax.set_xlabel('Betweenness centrality B')
ax.set_ylabel('Number of nodes having betweenness B')  


print(f'maximum closeness centrality = {max(closeness_centr)}')
print(f'minimum closeness centrality = {min(closeness_centr)}')
#note that the difference between max and min is pretty small

# SMALL-WORLD EFFECT
# An interesting property of entire networks, which is related to the closeness centrality, is the
# mean geodesic distance between vertices. In Section 8.2 we will use measurements of mean
# distance in networks to study the so-called “small-world effect.”

# i want to evaluate the mean geodesic distance to show the small world effect.
mean_geodesic_path = h.average_path_length(directed=False, unconn=True)
print(f'The mean geodesic path length is = {mean_geodesic_path}')
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




#%% COMPARING THE CENTRALITIES CORRELATIONS FOR THE UNDIRECTED UNWEIGHTED GRAPH
'''

COMPARING THE CENTRALITIES CORRELATIONS FOR THE WHOLE GRAPH

by alessio,  2021.10.05

in this section i will plot one centrality against the other to see their 
correlation and which behaviours may arise.

'''

def plt_compare_centralities(x, y, x_label, y_label, size):
    fig, ax = plt.subplots()
    ax.scatter(x, y, size) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

s = 0.4    # this is the size of the points of the scattergraph

plt_compare_centralities(degree_centr, between_centr, 
                         "Degree", "Betweenness", s)


plt_compare_centralities(between_centr, degree_centr, 
                         "Betweenness", "Degree", s)

plt_compare_centralities(closeness_centr, degree_centr, 
                         "Closeness", "Degrees", s)
#it's almost a gaussian lol

plt_compare_centralities(eigen_centr, degree_centr,
                         "Eigenvector centrality",
                         "Degrees", s)
# same as above but a little bit shifted to the left

plt_compare_centralities(closeness_centr, harm_centr,
                         "Closeness",
                         "Harmonic centrality", s)
 #they are mostly the same, as i expect



plt.scatter(eigen_centr, degree_centr, s) 
plt.scatter(closeness_centr, eigen_centr, s) 
plt.scatter(harm_centr, eigen_centr, s) 

plt.scatter(between_centr, degree_centr, s) 
plt.scatter(between_centr, closeness_centr, s) 
plt.scatter(between_centr, harm_centr, s) 

fig, ax = plt.subplots()
ax.scatter(between_centr, eigen_centr, 3) 
ax.set_xlabel('Betweenness')
ax.set_ylabel('Eigenvector centrality')
ax.set_xlim([0,600000])
ax.set_ylim([0,0.6])
ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

#%%% IGRAPH CENTRALITY MEASURES FOR ANTIVAX,NEUTRAL,PROVAX IN THE UNDIRECTED NETWORK

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
                      title='Degrees in logarithmic binning', 
                      xlabel='Degree k',
                      ylabel='Fraction of nodes having degree k')  

# compare the weighted degrees for the three categories
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').strength(weight="weight"),
                      h.vs.select(user_annotation='ProVax').strength(weight="weight"),
                      h.vs.select(user_annotation='Neutral').strength(weight="weight")),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Degrees in logarithmic binning')  

# compare the betweenness for the three categories with log binning
# n.b. the calculation of betweenness is the slowest
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='ProVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='Neutral').betweenness(weight="weight")),
                    legend=('AntiVax', 'ProVax', 'Neutral'),
                    title='Betwenness centrality in logarithmic binning')     

# compare the betweenness for the three categories with log y axis
# n.b. the calculation of betweenness is the slowest
compare_hist((h.vs.select(user_annotation='AntiVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='ProVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='Neutral').betweenness(weight="weight")),
                    legend=('AntiVax', 'ProVax', 'Neutral'),
                    title='Betwenness centrality with log y axis',
                    x_scale_log=False, y_scale_log=True)     


# compare the closeness for the three categories with normal histogram
compare_hist((h.vs.select(user_annotation='AntiVax').closeness,
                      h.vs.select(user_annotation='ProVax').closeness(weight="weight"),
                      h.vs.select(user_annotation='Neutral').closeness(weight="weight")),
             legend=('AntiVax', 'ProVax', 'Neutral'),
             title='Weighted closeness centrality')   
 
# compare the harmonic centrality for the three categories with normal histogram
#this does not work because harmonic centrality is not recognized wtf??
# well we can live without it since it is just a more sophisticated closeness..
compare_hist((h.vs.select(user_annotation='AntiVax').harmonic_centrality(weight="weight"),
                      h.vs.select(user_annotation='ProVax').harmonic_centrality(weight="weight"),
                      h.vs.select(user_annotation='Neutral').harmonic_centrality(weight="weight")),
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



#%%% CENTRALITY MEASURES NETWORKX 
#ON THE WHOLE GRAPH AS A WEIGHTED (DIRECTED OR NOT) NETWORK WITH NETWORKX

#                    WEIGHTED IN DEGREE

#i evaluate the IN_DEGREE (n.b. networkx returns it as an iterator so i have 
# to convert it in a list to pass it to my function that has to sort it)
# also the for loop is to save only the second element of the list because
# the networkx function returns a tuple (node_name, degree)
weighted_in_degree_centr = [x[1] for x in list(G.in_degree(weight="weight"))]
#log binning plot
weighted_in_alpha = plot_log_bin_hist(weighted_in_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Weighted in-degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Weighted in degree alpha = {weighted_in_alpha}')


########### 

#                   WEIGHTED OUT DEGREE

# see "IN DEGREE" for info
weighted_out_degree_centr = [x[1] for x in list(G.out_degree(weight="weight"))]
#log binning plot
weighted_out_alpha = plot_log_bin_hist(weighted_out_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Weighted out-degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Weighted out degree alpha = {weighted_out_alpha}')

##############

#                   WEIGHTED (UNDIRECTED) DEGREE

# see "IN DEGREE" for info
weighted_degree_centr = [x[1] for x in list(G.degree(weight="weight"))]
#log binning plot
weighted_alpha = plot_log_bin_hist(weighted_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Weighted (undirected) degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Weighted (undirected) degree alpha = {weighted_alpha}')

#################

#                    UNWEIGHTED IN DEGREE

#
unweighted_in_degree_centr = [x[1] for x in list(G.in_degree(weight=None))]
#log binning plot
unweighted_in_alpha = plot_log_bin_hist(unweighted_in_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Unweighted in-degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Unweighted in degree alpha = {unweighted_in_alpha}')


########### 

#                   UNWEIGHTED OUT DEGREE

# see "IN DEGREE" for info
unweighted_out_degree_centr = [x[1] for x in list(G.out_degree(weight=None))]
#log binning plot
unweighted_out_alpha = plot_log_bin_hist(unweighted_out_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Unweighted out-degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Unweighted in degree alpha = {unweighted_out_alpha}')

##############

#                   UNWEIGHTED (UNDIRECTED) DEGREE

# see "IN DEGREE" for info
unweighted_degree_centr = [x[1] for x in list(G.degree(weight=None))]
#log binning plot
unweighted_alpha = plot_log_bin_hist(unweighted_degree_centr, exp_start=0, exp_stop=3, number_of_bins=20,
                          degree_min=15,
                          fit=True, use_analytical_alpha=False,
                          title="Unweighted (undirected) degrees histogram with log binning",
                          xlabel='Degree k',
                          ylabel='Number of nodes having degree k\n normalized by bin width')

print(f'Unweighted (undirected) degree alpha = {unweighted_alpha}')


#############




############

#                WEIGHTED DIRECTED BETWEENNESS CENTRALITY


weighted_directed_betweenness = nx.betweenness_centrality(G,
                                                          weight="weight",
                                                          seed=338)

################

#                UNWEIGHTED DIRECTED BETWEENNESS CENTRALITY


unweighted_directed_betweenness = nx.betweenness_centrality(G,  
                                                            weight=None,
                                                          seed=339)


unweighted_directed_betweenness = list(nx.betweenness_centrality(G, 
                                                                 normalized=True,
                                                                 weight="weight",
                                                                 seed=338).values())
# plt.hist(unweighted_directed_betweenness, bins=20)
# there are really a lot (3705-751=2954 exactly) of assholes with 0 betweenness, now i will plot the histogram without them
u_d_b_norm = []
for i in range(len(unweighted_directed_betweenness)):
    if unweighted_directed_betweenness[i] > 0:
        u_d_b_norm.append(unweighted_directed_betweenness[i])
        
# plt.hist(u_d_b_norm, bins=20)

fig1, ax1 = plt.subplots()
ax1.hist(u_d_b_norm, bins=10)
ax1.set_title("Betweenness (un-normalized)")
plt.show()

# MAX BETWEENESS (UN-NORMALIZED) = 476076.34442694514  or 460713.0586046173 (unweighted)
# MIN BETWEENNESS : 0 (2954 USERS HAVE 0 BETWEENNESS), right after:
   # 0.25 (weighted) or 0.14285714285714285 (unweighted)

# IT SEEMS THAT CHANGIN THE WEIGHT OR THE SEED DOES NOT CHANGE THE HISTOGRAM AT ALL
# cioe no, qualcosa cambia, ma pochissimo, sembra che nell'istogramma i valori
# si shiftino di pochissimissimo ma la forma rimane esattamente uguale... boh!!
############

#                 EIGENVECTOR CENTRALITY


W_eigenvector_centr = list(nx.eigenvector_centrality(G, weight='weight').values())
plt.hist(np.log(W_eigenvector_centr), bins=20)


############

#                INWARD CLOSENESS CENTRALITY

#note that there is a difference in directed and undirected closeness
# this in particular is the inward closeness centrality, meaning

W_in_closeness_centr = list(nx.closeness_centrality(G, distance="weight").values())
plt.hist((W_in_closeness_centr), bins=50)



############

#                OUTWARD CLOSENESS CENTRALITY

# this is the ouward closeness centrality, meaning
W_out_closeness_centr = list(nx.closeness_centrality(G.reverse(), distance="weight").values())
plt.hist((W_out_closeness_centr), bins=50)


############

#                UNDIRECTED CLOSENESS CENTRALITY

W_und_closeness_centr =  list(nx.closeness_centrality(H, distance="weight").values())
plt.hist((W_und_closeness_centr), bins=50)



############


#                HARMONIC CENTRALITY

#
# 

W_harm_centr = list(nx.harmonic_centrality(G, distance="weight").values())
plt.hist((W_harm_centr), bins=50)



#%% NETWORKX COMPARING CENTRALITIES FOR THE THREE GROUPS

'''
NETWORKX
CENTRALITY MEASURES FOR ANTIVAX,NEUTRAL,PROVAX

 by alessio,  2021.10.13
 

in this section i want to evaluate different centrality measures on the
three groups: antivax, novax, neutrals and compare them by plotting 
them as logarithmic binned histograms.

 
'''



#I can use the G.subgraph(selected_nodes) function to access the nodes of just one group

antivax_nodes = [n for n,v in G.nodes(data=True) if v['user_annotation'] == 'AntiVax']  

# compare the degrees for the three categories
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').degree(weight="weight"),
                      h.vs.select(user_annotation='ProVax').degree(weight="weight"),
                      h.vs.select(user_annotation='Neutral').degree(weight="weight")),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Degrees in logarithmic binning')  

# compare the weighted degrees for the three categories
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').strength(weight="weight"),
                      h.vs.select(user_annotation='ProVax').strength(weight="weight"),
                      h.vs.select(user_annotation='Neutral').strength(weight="weight")),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Degrees in logarithmic binning')  

# compare the betweenness for the three categories with log binning
# n.b. the calculation of betweenness is the slowest
compare_log_bin_hist((h.vs.select(user_annotation='AntiVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='ProVax').betweenness(weight="weight"),
                      h.vs.select(user_annotation='Neutral').betweenness(weight="weight")),
                    legend=('AntiVax', 'ProVax', 'Neutral'),
                    title='Betwenness centrality in logarithmic binning')     

# compare the betweenness for the three categories with log y axis
# n.b. the calculation of betweenness is the slowest
compare_hist((h.vs.select(user_annotation='AntiVax').betweenness(),
                      h.vs.select(user_annotation='ProVax').betweenness(),
                      h.vs.select(user_annotation='Neutral').betweenness()),
                      legend=('AntiVax', 'ProVax', 'Neutral'),
                      title='Betwenness centrality with log y axis',
                      x_scale_log=False, y_scale_log=True, 
                      x_label="Betweenness centrality C", 
                      y_label="Number of nodes with betweenness C")     


# compare the closeness for the three categories with normal histogram
compare_hist((h.vs.select(user_annotation='AntiVax').closeness(weight="weight"),
                      h.vs.select(user_annotation='ProVax').closeness(weight="weight"),
                      h.vs.select(user_annotation='Neutral').closeness(weight="weight")),
             legend=('AntiVax', 'ProVax', 'Neutral'),
             title='Weighted closeness centrality')   
 
# compare the harmonic centrality for the three categories with normal histogram
#this does not work because harmonic centrality is not recognized wtf??
# well we can live without it since it is just a more sophisticated closeness..
compare_hist((h.vs.select(user_annotation='AntiVax').harmonic_centrality(weight="weight"),
                      h.vs.select(user_annotation='ProVax').harmonic_centrality(weight="weight"),
                      h.vs.select(user_annotation='Neutral').harmonic_centrality(weight="weight")),
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


#%% WHO HAS THE HIGHEST CENTRALITY etc?



#i save all the user_ids in order in a list, they will be useful 
user_ids = [x[0] for x in list(G.nodes(data="user.id"))]

# WHO HAS THE HIGHEST DEGREE?


# it could be funny to see if the user with the highest degree, betweenness ecc 
# is a provaxxers, novaxxers etc.
# i save in a list the list of all the degrees sorted from highest to lowest


def sort_by_centrality(centrality_to_sort, nodes, centr_name):
    
    '''
        
    this function sorts by centrality and gives the users ordered from highest
    to lowest centrality (depending on which centrality you give to sort)
    and also prints the user annotation of the top 5 users and also
    gives the position and centrality of the 1st antivax, 1st provax and 1st neutral
    ordered by centrality
    
    INPUTS:
        
        centrality_to_sort : list of centrality measure, example: G.degree 
        nodes : (networkx?) node list (G.nodes)
        centr_name : string, the name of the centrality just for printing it nicely
        
    OUTPUTS: 
        
        centr_high_to_low : list of users (nodes) sorted by the desired centrality
        
    
    '''
    centr_high_to_low = sorted(centrality_to_sort, key=lambda x: x[1], reverse=True)
    
    # i create a list of strings (i know there are more elegant ways but shut up)
    highest_centr_annotations = ["a", "b", "c", "d", "e"]
    # i check for the first five which user_annotation they have
    for (p,d) in nodes(data="user_annotation"):
         if p == centr_high_to_low[0][0]:
             
             
             highest_centr_annotations[0] = d
         if p == centr_high_to_low[1][0]:
             highest_centr_annotations[1] = d
         if p == centr_high_to_low[2][0]:
             highest_centr_annotations[2] = d
         if p == centr_high_to_low[3][0]:
             highest_centr_annotations[3] = d
         if p == centr_high_to_low[4][0]:
             highest_centr_annotations[4] = d       
             
    print(f'the first five users with highest {centr_name} are:')
    print(f'{highest_centr_annotations[0]}: {centr_high_to_low[0][1]}, user_id: {centr_high_to_low[0][0]}')
    print(f'{highest_centr_annotations[1]}: {centr_high_to_low[1][1]}, user_id: {centr_high_to_low[1][0]}')
    print(f'{highest_centr_annotations[2]}: {centr_high_to_low[2][1]}, user_id: {centr_high_to_low[2][0]}')
    print(f'{highest_centr_annotations[3]}: {centr_high_to_low[3][1]}, user_id: {centr_high_to_low[3][0]}')
    print(f'{highest_centr_annotations[4]}: {centr_high_to_low[4][1]}, user_id: {centr_high_to_low[4][0]}')
     
    
    # WHO IS THE PROVAXXERS WITH THE HIGHEST DEGREE?
    # let's see just for fun at which place the provaxxer with highest degrees ranks
    found=False   # this is a flag that i will use to stop the loop once i've found the first one
    for i in range(len(nodes)):
        for (p,d) in nodes(data="user_annotation"):
            if p == centr_high_to_low[i][0]:
                if d == "ProVax":
                     print(f'first provaxxer at position {(i+1)} with {centr_high_to_low[i][1]}, user_id: {centr_high_to_low[i][0]}')
                     found=True
                     break
        if found == True:
            break   
    # WHO IS THE ANTVAXXERS WITH THE HIGHEST DEGREE?
    # let's see just for fun at which place the provaxxer with highest degrees ranks
    found=False   # this is a flag that i will use to stop the loop once i've found the first one
    for i in range(len(nodes)):
        for (p,d) in nodes(data="user_annotation"):
            if p == centr_high_to_low[i][0]:
                if d == "AntiVax":
                     print(f'first antivaxxers at position {(i+1)} with {centr_high_to_low[i][1]}, user_id: {centr_high_to_low[i][0]}')
                     found=True
                     break
        if found == True:
            break                       
    # WHO IS THE NEUTRAL WITH THE HIGHEST DEGREE?
    # let's see just for fun at which place the provaxxer with highest degrees ranks
    found=False   # this is a flag that i will use to stop the loop once i've found the first one
    for i in range(len(nodes)):
        for (p,d) in nodes(data="user_annotation"):
            if p == centr_high_to_low[i][0]:
                if d == "Neutral":
                     print(f'first neutral at position {(i+1)} with {centr_high_to_low[i][1]}, user_id: {centr_high_to_low[i][0]}')
                     found=True
                     break
        if found == True:
            break                       


    return centr_high_to_low


centrality_to_sort =  G.in_degree(weight=None)
centr_name = 'unweighted in degree'

''' we can sort:
    
    unweighted undirected degree: G.degree(weight=None)
    weighted undirected degree: G.degree(weight='weight')
    unweighted in degree: G.in_degree(weight=None)
    weighted in degree: G.in_degree(weight='weight')
    unweighted out degree: G.out_degree(weight=None)
    weighted out degree: G.out_degree(weight='weight')
    betwenness :  nx.betweenness_centrality(G,
                                              normalized=False,
                                              weight='weight',
                                              seed=338).items()
    
    
    etc...

 
'''


# this function sorts by centrality and gives the users ordered from highest
# to lowest centrality (depending on which centrality you give to sort)
# and also prints the user annotation of the top 5 users and also
# gives the position and centrality of the 1st antivax, 1st provax and 1st neutral
# ordered by centrality
centr_high_to_low = sort_by_centrality(centrality_to_sort, G.nodes, centr_name)





# # let's do it for betweenness - remember that nodes with high betweenness
# # are brokers, they control the information flux



# betweenness_high_to_low = sorted(list(zip(user_ids, W_between_centr)),
#                                  key=lambda x: x[1], reverse=True)

# highest_betweenness_annotations = ["a", "b", "c", "d", "e"]
# for (p,d) in G.nodes(data="user_annotation"):
#      if p == betweenness_high_to_low[0][0]:
#          highest_betweenness_annotations[0] = d
#      if p == betweenness_high_to_low[1][0]:
#          highest_in_degrees_annotations[1] = d
#      if p == betweenness_high_to_low[2][0]:
#          highest_betweenness_annotations[2] = d
#      if p == betweenness_high_to_low[3][0]:
#          highest_betweenness_annotations[3] = d
#      if p == betweenness_high_to_low[4][0]:
#          highest_betweenness_annotations[4] = d       

# # the result is:
# #     ['AntiVax', 'AntiVax', 'AntiVax', 'AntiVax', 'Neutral']

# # let's see just for fun at which place the provaxxer with highest betweenners ranks
# found=False   # this is a flag that i will use to stop the loop once i've found the first one
# for i in range(len(G.nodes)):
#     for (p,d) in G.nodes(data="user_annotation"):
#         if p == betweenness_high_to_low[i][0]:
#             if d == "ProVax":
#                  print(r"first provaxxer at position "+str(i+1))
#                  found=True
#                  break
#     if found == True:
#         break        
# # result:      first provaxxer in order of betweenness at position 14
    
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
# the assirtativity coefficient is 0.08819015433238132, which is positive =>
#  people tend to connect with people of the same group 



#%% communities detection with LOUVAIN algorithm
'''

by alessio 

'''

if LOUVAIN == True:
    # finding the communities with the louvain algorithm
    communities = h.community_multilevel(weights=df["weight"], return_levels=False)
    
    
    print("number of communities found:")
    print(len(communities))



#%% communtiies detection with LEIDEN algorithm
'''

communtiies detection with leiden algorithm
which should be a better versoin of the louvain algorithm
but i mostly choose it because it has more parameter that we can fine tune

also it can be used to improve the estimate on an already known membership,
so we can use it for example feeding into it the result of another algorithm
(or even the antivax, provax, neutral groups) and see how it improves the result

for more info: 
    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). 
    From Louvain to Leiden: guaranteeing well-connected communities.
    Scientific reports, 9(1), 5233. doi: 10.1038/s41598-019-41695-z
    
by alessio 2021.10.27 21.31

'''
# DIRECTED_COMMUNITY_DETECTION = True
# WEIGHTED_COMMUNITY_DETECTION = True

# this block here is if you want to use the old version of the algorithm


# if DIRECTED_COMMUNITY_DETECTION == False and WEIGHTED_COMMUNITY_DETECTION == False:
    
#     # finding the communities with the louvain algorithm
#     communities = h.community_leiden(objective_function='modularity', 
#                                      weights=None,
#                                      resolution_parameter=0.3,
#                                      beta=0.01,
#                                      initial_membership=None,
#                                      n_iterations=-1)
#     #objective_function can either be CPM or modularity.
#     # weights are edge weights
#     # resolution_paramater is a resolution parameter lol meaning that lower
#     # resolution leads to fewer and larger communities, high resolution instead 
#     # yields more and smaller communiteis.
#     # beta parameter affecting the randomness in the Leiden algorithm.
#                   # don't know what that means, i will leave it at the default 0.01
#     # initial_membership is if we want to pass a known membership to refine it
#     # n_iterations is number of iterations; if you use a negative number then 
#     # it will iterate until a iteration does not improve the measure

 
    
# elif DIRECTED_COMMUNITY_DETECTION == False and WEIGHTED_COMMUNITY_DETECTION == True:
    
#     communities = h.community_leiden(objective_function='modularity', 
#                                      weights=h.es['weight'],
#                                      resolution_parameter=0.3,
#                                      beta=0.01,
#                                      initial_membership=None,
#                                      n_iterations=-1)
        
seed = 338
n_iterations=1
max_comm_size=0

if WEIGHTED_COMMUNITY_DETECTION == False:
    
    # this is another (newer) algorithm of the leiden method 
    # just as the other one, this is also made by v.traag the author of the
    # paper of the leiden algorithm, so it is just as good
    # anyway this has implemented the possibility of working with directed graphs
    communities = la.find_partition(h, 
                                    partition_type=la.ModularityVertexPartition,
                                    initial_membership=None,
                                    weights=None,
                                    n_iterations=n_iterations, 
                                    max_comm_size=max_comm_size,
                                    seed=seed)

elif WEIGHTED_COMMUNITY_DETECTION == True:
    
    communities = la.find_partition(h, 
                                    partition_type=la.ModularityVertexPartition,
                                    initial_membership=None,
                                    weights=h.es['weight'],
                                    n_iterations=n_iterations, 
                                    max_comm_size=max_comm_size,
                                    seed=seed)

else: 
    print("UNDIRECTED_COMMUNITY_DETECTION and WEIGHTED_COM.. must be either True or False")
    
    
print("number of communities found:")
print(len(communities))

membership = communities.membership

#%% communities detection but instead of using the algorithm once i 
# run it many times and then combine the results




if AVERAGE_ALGORITHM == True:
    runs = 5
    
    membershipping = np.zeros((runs, len(h.vs)))
    
    for i in range(runs):
        if WEIGHTED_COMMUNITY_DETECTION == False:
            
            # this is another (newer) algorithm of the leiden method 
            # just as the other one, this is also made by v.traag the author of the
            # paper of the leiden algorithm, so it is just as good
            # anyway this has implemented the possibility of working with directed graphs
            communities = la.find_partition(h, 
                                            partition_type=la.ModularityVertexPartition,
                                            initial_membership=None,
                                            weights=None,
                                            n_iterations=n_iterations, 
                                            max_comm_size=max_comm_size,
                                            seed=seed)
        
        elif WEIGHTED_COMMUNITY_DETECTION == True:
            
            communities = la.find_partition(h, 
                                            partition_type=la.ModularityVertexPartition,
                                            initial_membership=None,
                                            weights=h.es['weight'],
                                            n_iterations=n_iterations, 
                                            max_comm_size=max_comm_size,
                                            seed=seed)
        
        else: 
            print("UNDIRECTED_COMMUNITY_DETECTION and WEIGHTED_COM.. must be either True or False")
        
        
        
        membershipping[i] = communities.membership
        print(f"run #{i}, seed={seed}, number of communities found: {len(communities)}")
        seed = seed + 1
    
    
    
    from collections import Counter
    
    average_membership = np.zeros(len(h.vs))
    for i in range(len(h.vs)): 
        average_membership[i] = Counter(membershipping[:,i]).most_common(1)[0][0]
    
    
    membership = average_membership
     
    print("average completed, number of communities found:")
    print(len(communities))
    
    
    # CI STA UNA CAZZO DI FALLA IN QUESTO CAZZO DI CODICE NON FA IL CAZZO DI NIENTE 
    # DI AVERAGE MANCO PER IL CAZZO PORCA MADONNA PORCA MADONNA
    # PORCA MADONNA PORCA MADONNA
    
    # DEVO SOLO ACCETTARE CHE CI STA UNA DIFFERENZA DI MAX 5%
    
    

#%% visualizing the communities
''' by alessio '''


def plot_communities(h, communities, file_name, vertex_size=5, vertex_opacity='FF'):
    
    ''' by alessio 
    
    h, igraph.Graph object
    communities, igraph.VertexClustering object
    file_name, str: the name of the file to which the plot is saved (include the directory)
    vertex_size, int (or even float?): size of vertices in the plot
    vertex_opacity, HEX:  WARNING, must be a 2 digit hexadecimal number or string
    
    '''
    
    # SETTING NODES COLOR
    #creating a color map : a list of n random color codes, where n is the number of communities 
    cmap = ['#'+''.join([random.choice('0123456789abcdef') for x in range(6)]) for z in range(len(communities))]
    cmap = [str(color)+str(vertex_opacity) for color in cmap]
    # saving the colors in a list; this list uses the same index of the nodes list,
    # so the j-th color corresponds to the j-th node. 
    #Different colors are assigned to member of different communities.
    vcolors = {v: cmap[i] for i, c in enumerate(communities) for v in c}
    # assigning colors from vcolors to nodes as an attribute
    h.vs["color"] = [vcolors[v] for v in h.vs.indices]
    
    # # SETTING NODES SHAPE
    
    # smap = ['rectangle', 'circle', 'triangle-up', 'diamond']
    # vshapes = {v: smap[i] for i, c in enumerate(h.vs["user_annotation"]) for v in c}
    # h.vs['shape'] = [vshapes[v] for v in h.vs.indices]
    
    
    
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
    eweights = {e.index: (1*h.vcount()) if member[e.tuple[0]]==member[e.tuple[1]] else 0.1 for e in h.es}
    # assigning colors from eweights to edges as an attribute
    h.es["weight"] = [eweights[e.index] for e in h.es]
    
    
    # the visual_style dict stores the attribute to pass to the plot function to control the visual appereance of the plot
    visual_style = {}
    visual_style["vertex_size"] = vertex_size
#    visual_style["vertex_shape"] = 
    visual_style["edge_color"] = '#00009001'
    visual_style["vertex_frame_width"] = 0
    visual_style["layout"] = h.layout_fruchterman_reingold(weights=h.es["weight"])
    visual_style["bbox"] = (1024, 1024)
    visual_style["margin"] = 20
    
    #plotting to file 
    ig.plot(h, file_name, mark_groups=True, **visual_style)
    
    return 


            
if visualize_communities == True:     
    plot_communities(h, communities,
          file_name="D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_test_2022.07.18.18.26.png")

# plot_communities(h, communities,
#       file_name="D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_testtt.png")





# def plot_communities2(h, communities, file_name, vertex_size=3):
    
#     ''' by alessio 
    
#        in this function i was trying to simply color code the nodes on their user 
#        annotation but for some reason i failed
      
#     h, igraph.Graph object
#     communities, igraph.VertexClustering object
#     file_name, str: the name of the file to which the plot is saved (include the directory)
#     vertex_size, int (or even float?): size of vertices in the plot
    
#     '''
    
#     # SETTING NODES COLOR depending on the user_annotation     
#     user_annotation_groups = [[],[],[],[]]   # antivax, provax, neutral, N/A
#     i=0
#     for node in h.vs:
#         if node["user_annotation"]=="AntiVax":
#             user_annotation_groups[0].append(i)
#         if node["user_annotation"]=="ProVax":
#             user_annotation_groups[1].append(i)
#         if node["user_annotation"]=="Neutral":
#                 user_annotation_groups[2].append(i)
#         if node["user_annotation"]=="N/A":
#                 user_annotation_groups[3].append(i)
#         i=i+1

#     cmap = ['#FF9933', '#66FF99', '#BAC761', '#F5F5F5']
#     vcolors = {v: cmap[i] for i, c in enumerate(user_annotation_groups) for v in c}
#     # assigning colors from vcolors to nodes as an attribute
#     h.vs["color"] = [vcolors[v] for v in h.vs.indices]
#     # # SETTING NODES SHAPE
    
#     # smap = ['rectangle', 'circle', 'triangle-up', 'diamond']
#     # vshapes = {v: smap[i] for i, c in enumerate(h.vs["user_annotation"]) for v in c}
#     # h.vs['shape'] = [vshapes[v] for v in h.vs.indices]
    
    
    
#     # SETTING EDGES COLOR
    
#     # # saving the membership of each node (a list of which community each node belongs to)
#     member = communities.membership
#     # # saving the colors in a list; this list uses the same index of the edges list,
#     # # so the j-th color corresponds to the j-th edge. 
#     # # all the edges within a community use the same colour. Edges between communities are grey 
#     # ecolors = {e.index: cmap[member[e.tuple[0]]] if member[e.tuple[0]]==member[e.tuple[1]] else "#e0e0e0" for e in h.es}
#     # # assigning colors from ecolors to edges as an attribute
#     # h.es["color"] = [ecolors[e] for e in h.es.indices]
   
#     # SETTING EDGES WEIGHT  - wait, does this actually cancel the information on weight? should i use edge_width instead?
    
#     # same as above but for weights. edges within a community have a much higher line weight
#     eweights = {e.index: (2*h.vcount()) if member[e.tuple[0]]==member[e.tuple[1]] else 0.1 for e in h.es}
#     # assigning colors from eweights to edges as an attribute
#     h.es["weight"] = [eweights[e.index] for e in h.es]
    
    
#     # the visual_style dict stores the attribute to pass to the plot function to control the visual appereance of the plot
#     visual_style = {}
#     visual_style["vertex_size"] = vertex_size
# #    visual_style["vertex_shape"] = 
#     visual_style["edge_color"] = '#00009001'
#     visual_style["vertex_frame_width"] = 0
#     visual_style["layout"] = h.layout_fruchterman_reingold(weights=h.es["weight"])
#     visual_style["bbox"] = (1024, 1024)
#     visual_style["margin"] = 10
#     #plotting to file 
#     ig.plot(h, file_name, mark_groups=True, **visual_style)
    
#     return 

       

# plot_communities2(h, communities,
#       file_name="D:/Users/aless/Desktop/Universita/Complex_Networks_Remondini/Twitter_proj/test_plots/ig_testtt4.png")

#%% VISUALIZING COMMUNITIES V2

'''

VISUALIZING COMMUNITIES V2

THIS DOES NOT WORK

FROM : https://graphsandnetworks.com/community-detection-using-networkx/

'''


if visualize_communities_2 == True: 
        
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (25, 10)})
    # get reproducible results
    from numpy import random as nprand
    random.seed(123)
    nprand.seed(123)
    
    def set_node_community(G, communities):
          '''Add community to node attributes'''
          for c, v_c in enumerate(communities):
              for v in v_c:
                  # Add 1 to save 0 for external edges
                  G.nodes[v]['community'] = c + 1
                  
    def set_edge_community(G):
        '''Find internal edges and add their community to their attributes'''
        for v, w, in G.edges:
            if G.nodes[v]['community'] == G.nodes[w]['community']:
                # Internal edge, mark with community
                G.edges[v, w]['community'] = G.nodes[v]['community']
            else:
                # External edge, mark as 0
                G.edges[v, w]['community'] = 0
                
    def get_color(i, r_off=1, g_off=1, b_off=1):
        '''Assign a color to a vertex.'''
        r0, g0, b0 = 0, 0, 0
        n = 16
        low, high = 0.1, 0.9
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)
      
    pos = nx.spring_layout(G, k=0.1)
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (25, 10)})
    plt.style.use('white_background')
    
    communityes = communities.as_clustering()
    communityes = sorted(communityes)
    # Set node and edge communities
    set_node_community(G, communityes)
    set_edge_community(G)
    
    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    
    # external edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edgelist=external,
        edge_color="silver",
        node_color=node_color,
        alpha=0.2,
        with_labels=False)
    # internal edges
    nx.draw_networkx(
        G, pos=pos,
        edgelist=internal,
        edge_color=internal_color,
        node_color=node_color,
        alpha=0.05,
        with_labels=False)


#%% VISUALIZING COMMUNITIES V3

'''

VISUALIZING COMMUNITIES V3

THIS DOES NOT WORK

FROM : https://igraph.org/python/tutorial/develop/tutorials/visualize_communities/visualize_communities.html

'''

#communities = communities.as_clustering()
# Color each vertex and edge based on its community membership
num_communities = len(communities)
palette = ig.RainbowPalette(n=num_communities)
for i, community in enumerate(communities):
    h.vs[community]["color"] = i
    # community_edges = h.es.select(_within=community)
    # community_edges["color"] = i
    
   
    
    # Plot with only vertex and edge coloring
fig, ax = plt.subplots()
ig.plot(
    communities,
    palette=palette,
    edge_width=0.01,
    target=ax,
    vertex_size=5,
    edge_arrow_size=0
)

legend_handles = []
for i in range(num_communities):
    handle = ax.scatter(
        [], [],
        s=100,
        facecolor=palette.get(i),
        edgecolor="k",
        label=i,
    )
    legend_handles.append(handle)

ax.legend(
    handles=legend_handles,
    title='Community:',
    bbox_to_anchor=(0, 1.0),
    bbox_transform=ax.transAxes,
)

#%%

''' visualize communities V4

THIS WORKS

'''


ig.plot(communities, 
        vertex_size=5,
        vertex_frame_width=0,
        edge_width=0.1,
        edge_arrow_size=0)

legend_handles = []
for i in range(num_communities):
    handle = ax.scatter(
        [], [],
        s=100,
        facecolor=palette.get(i),
        edgecolor="k",
        label=i,
    )
    legend_handles.append(handle)
plt.legend(
    handles=legend_handles,
    title='Community:',
    bbox_to_anchor=(0, 1.0),
    bbox_transform=ax.transAxes,
)

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
        memberships_antivax.append(membership[i])
    elif h.vs['user_annotation'][i] == 'ProVax':
        memberships_provax.append(membership[i])
    elif h.vs['user_annotation'][i] == 'Neutral':
        memberships_neutral.append(membership[i])
    else:
        memberships_non_annotated.append(membership[i])
    
# histograms of memberships to communities

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
    ax.set_yscale('log')
    if WEIGHTED_COMMUNITY_DETECTION == True:
        weighted = 'Weighted'
    else:
        weighted = 'Unweighted'
    if DIRECTED_COMMUNITY_DETECTION == True:
        directed = 'directed'
    else:
        directed = 'undirected'
    plt.title(f"{weighted} and {directed} network")
    plt.show()
    
#%% PIE CHARTS

'''
i should do something like: for each community i should show what's
the percentage of antivax, provax, neutral.

'''


anti_count = np.zeros(len(communities))
neut_count = np.zeros(len(communities))
pro_count = np.zeros(len(communities))
total_count = np.zeros(len(communities))
i=0
for subgraph in communities.subgraphs():
    anti_count[i]=sum(node["user_annotation"]=="AntiVax" for node in subgraph.vs)
    neut_count[i]=sum(node["user_annotation"]=="Neutral" for node in subgraph.vs)
    pro_count[i]=sum(node["user_annotation"]=="ProVax" for node in subgraph.vs)
    total_count[i]=sum(True for node in subgraph.vs)
    i = i+1


anti_count_total=sum(anti_count)
neut_count_total=sum(neut_count)
pro_count_total=sum(pro_count)
total_count_total=sum(total_count)



def plot_pie_chart_community(i, communities, anti_count, pro_count, neut_count,
                             total_count, show_plot=True, save_plot=False, directory=None):

        ''' this simple function plots pie chart for the i-th community
        i'm writing this function just to plot many graphs in a clean way
        
        i    :  selects which community
        
        n.b. this is really a AD HOC function and it is not intended to be
        used elsewhere, for this reason i'm not documenting it properly
        
        '''

        labels = 'AntiVax', 'ProVax', 'Neutral', 'Non Annotated'
        sizes = [anti_count[i]*100/total_count[i],
                 pro_count[i]*100/total_count[i],
                 neut_count[i]*100/total_count[i],
                 (total_count[i]-pro_count[i]-neut_count[i]-anti_count[i])*100/total_count[i]]
        explode = (0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
        colors = '#FF9933', '#66FF99', '#BAC761', '#F5F5F5'
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=180)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title(r"Community "+str(i)+' - '+str(len(communities[i]))+' users')
        if show_plot == True:
            plt.show()
        if save_plot == True:
            if directory is None:
                print('directory for saving not specified')
            else:
                 filename = f'{directory}\\pie_comm_{i}.png'
                 plt.savefig(filename, bbox_inches='tight')
                 print(f'pie_comm_{i}.png saved')
        

for i in range(len(communities)):
    plot_pie_chart_community(i, communities, anti_count, pro_count, neut_count,
                             total_count, 
                             show_plot=False,
                             save_plot=False,
                             directory=r'D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj\pie_charts')

with open(r'D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj\pie_charts\readme.txt', 'w') as f:
    f.write(f'WEIGHTED = {WEIGHTED_COMMUNITY_DETECTION}\nDIRECTED = {DIRECTED_COMMUNITY_DETECTION}\nn_iterations = {n_iterations}\nseed = {seed}')

# pie chart plot for the whole graph

labels = 'AntiVax', 'ProVax', 'Neutral', 'Non Annotated'
sizes = [anti_count_total*100/total_count_total,
         pro_count_total*100/total_count_total,
         neut_count_total*100/total_count_total,
         (total_count_total-pro_count_total-neut_count_total-anti_count_total)*100/total_count_total]
explode = (0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = '#FF9933', '#66FF99', '#BAC761', '#F5F5F5'
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Whole Network")
plt.show()

#%% STACKED BAR PLOT of the communities

# this is like the previous pie charts, but all the communities on the same graph

# i_start and i_end are to select which communities to plot
i_start = 0 
i_end = 16
length = i_end - i_start

labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48']
labels = labels[i_start:i_end]

colors = '#FF9933', '#66FF99', '#BAC761', '#DEDEDE'

width = 0.40       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, anti_count[i_start:i_end], width, label='Anti', color=colors[0])
ax.bar(labels, pro_count[i_start:i_end], width, bottom=anti_count[i_start:i_end], label='Pro', color=colors[1])
ax.bar(labels, neut_count[i_start:i_end], width, bottom=anti_count[i_start:i_end]+pro_count[i_start:i_end], label='Neut', color=colors[2])
ax.bar(labels, (total_count-pro_count-neut_count-anti_count)[i_start:i_end], width, bottom=anti_count[i_start:i_end]+pro_count[i_start:i_end]+neut_count[i_start:i_end], label='N/A', color=colors[3])



ax.set_ylabel('# of users')
ax.set_xlabel('Community')
ax.set_title('Absolute quantities of users in communities')
ax.legend()

plt.show()





#if we want to plot percentages..
anti_percentages = np.zeros(len(communities))
pro_percentages = np.zeros(len(communities))
neut_percentages = np.zeros(len(communities))
na_percentages = np.zeros(len(communities))

# for i in range(len(labels)):
       
#     anti_percentages[i] = anti_count[i]*100/total_count[i]
#     pro_percentages[i] = pro_count[i]*100/total_count[i]
#     neut_percentages[i] = neut_count[i]*100/total_count[i]
#     na_percentages = 100 - anti_percentages[i] - pro_percentages[i] - neut_percentages[i]


anti_percentages = anti_count*100/total_count
pro_percentages = pro_count*100/total_count
neut_percentages = neut_count*100/total_count
na_percentages = (total_count-pro_count-neut_count-anti_count)*100/total_count

width = 0.40       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, anti_percentages[i_start:i_end], width, color=colors[0], label='AntiVax')
ax.bar(labels, pro_percentages[i_start:i_end], width, bottom=anti_percentages[i_start:i_end], color=colors[1], label='ProVax')
ax.bar(labels, neut_percentages[i_start:i_end], width, bottom=anti_percentages[i_start:i_end]+pro_percentages[i_start:i_end],color=colors[2], label='Neutral')
ax.bar(labels, na_percentages[i_start:i_end], width, bottom=anti_percentages[i_start:i_end]+pro_percentages[i_start:i_end]+neut_percentages[i_start:i_end],color=colors[3], label='Non-Annotated')



ax.set_ylabel('% of users')
ax.set_xlabel('Community')
ax.set_title('Relative quantities of users in communities')
ax.legend()

plt.show()


#%% pie chart on user annotation

# now i do the complementary to the previous section: i plot a pie chart of 
# different user annotation groups highlighting the different communities



# pie chart plot

def plot_pie_by_groups(communities, count, total_count, name="Users"):

    # this is a messy way to create a list of strings 0 to x
    labels = list(map(str, list(map(int, np.linspace(0,len(communities)-1, len(communities))))))
    
    
    sizes = count/total_count
    for i in range(len(labels)):
        if sizes[i]<0.01:
            labels[i] = None
            
    explode = np.ones(len(communities))*0.05 
    # colors = '#FF9933', '#66FF99', '#BAC761', '#F5F5F5'
    fig1, ax1 = plt.subplots()
    cmap = plt.cm.RdYlGn
    ax1.pie(sizes, explode=explode, labels=labels,
            shadow=False, startangle=90, 
            pctdistance=0.8, frame=False, 
            colors=[*cmap(random.sample(list(np.linspace(0,1,len(communities))), len(communities)))])
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(r""+name+" presence in the communities "' - '+str(int(sum(count)))+' users total')
    plt.show()

plot_pie_by_groups(communities, anti_count, total_count, "AntiVax")


plot_pie_by_groups(communities, pro_count, total_count, "ProVax")


plot_pie_by_groups(communities, neut_count, total_count, "Neutral")


plot_pie_by_groups(communities, total_count-(anti_count+pro_count+neut_count), 
                   total_count, "Non annotated")



#%% CENTRALITY MEASURES ON COMMUNITIES

'''

CENTRALITY MEASURES ON COMMUNITIES

 by Alessio , 2021.10.11



in this section i measure centrality measures for the different communities
found in the community detection section.
I compare those measures by plotting them as histograms with logarithm binning.
n.b in order to compare should i normalize by the number of vertices in the community?
yes. but it is already done by default by the igraph functions when computing
the centrality measures. is it??? from the picture it doesn't seem so
 
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




# for example, i plot on the same histogram the degrees of communities 7, 8 ,9
compare_log_bin_hist(degrees_coms[6:9],
                     legend=("Community 7", "Community 8","Community 9"),
                     title="Degrees in log binning for communities 7, 8 and 9")
                     


