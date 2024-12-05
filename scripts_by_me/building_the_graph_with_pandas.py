# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 00:32:16 2021


building the graph.
the simple and clean way with pandas (actually pandas is useful to neatly preprocess 
                                      the dataframe before creating the graph)


    ██████                        ██████    
  ██████████  ████████████████  ██████████  
██████████████                ██████████████
████████                            ████████
██████                                ██████
  ██                                    ██  
  ██                                    ██  
██        ██████            ██████        ██
██      ██████████        ██████████      ██
██    ████████  ██        ██  ████████    ██
██    ████████  ██        ██  ████████    ██
██    ██████████            ██████████    ██
██      ██████      ████      ██████      ██
  ██                ████                ██  
  ████████  ▒▒▒▒▒▒        ▒▒▒▒▒▒  ████████  
████████████▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒████████████
██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
  ████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████████████  
    ████████  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ████████    
                ▒▒▒▒▒▒▒▒▒▒▒▒                
                  ▒▒▒▒▒▒▒▒                  
                    ▒▒▒▒                    




@author: ALESSIO
"""


import pandas as pd
import networkx as nx


#%% Reading the dataset - it takes some time

#We will use the following path to read the data
datafolder_path = r""
datafile = datafolder_path + r"\df_annotated_2021-01-26.pkl"

#We read the datafile and make a dataframe
df = pd.read_pickle(datafile)


#%% creating the graph

# i remove from my dataframe the rows for which user_annotation is nan
# (we only want to keep the user with a user_annotation)
df = df.loc[~df["user_annotation"].isna()]

# i remove from my dataframe the rows for which "retweeted_status.id" is nan
# (we only want to keep the tweets that are retweets)
df = df.loc[~df["retweeted_status.id"].isna()]

# we count how many times user A retweeted user B and store the values in a "weight" attribute
df['weight'] = df.groupby(['user.id', 'retweeted_status.user.id'])['user.id'].transform('size')

# i create the networkx graph from the dataframe
G = nx.from_pandas_edgelist(df, 'user.id', 'retweeted_status.user.id',
                            create_using=nx.DiGraph(), edge_attr='weight')









#%% useful pandas functions to handle dataframes!!!!

"""

USEFUL PANDAS FUNCTIONS TO HANDLE DATAFRAMES
             
            ,,,         ,,,
          ;"   ^;     ;'   ",
          ;    s$$$$$$$s     ;
          ,  ss$$$$$$$$$$s  ,'
          ;s$$$$$$$$$$$$$$$
          $$$$$$$$$$$$$$$$$$
         $$$$P""Y$$$Y""W$$$$$
         $$$$  p"$$$"q  $$$$$
         $$$$  .$$$$$.  $$$$
          $$DcaU$$$$$$$$$$
by          "Y$$$"*"$$$Y"    aka
dcau            "$b.$$"     Panda



"""




# df.loc[(CONDITION1) & (CONDITION2)] to access only a subset of the dataframe
# for example:
df.loc[(df["user_annotation"] == "ProVax") & (df["annotation"] == "AntiVax")]
# for example i can collect all the user.ids of users with provax annotation but that posted a antivax annotated tweet
provax_user_antivax_tweet = df.loc[(df["user_annotation"] == "ProVax") & (df["annotation"] == "AntiVax")]["user.id"]


#with this i filter the rows that are not nan in retweeted_status.id
df.loc[~df["retweeted_status.id"].isna()]


# to REMOVE (i.e. CANCEL) columns from the dataframe
df = df.drop(columns=["user.id"])



# let's say i want to save in a new dataframe only a subset of the rows of the original dataframe
# for example, i might saved a new dataframe with only the tweets by annotated users:
df_annotated_users = df.loc[~df["user_annotation"].isna()]

df_annotated_users.to_csv(r'D:\Users\aless\Desktop\Universita\Complex_Networks_Remondini\Twitter_proj\df_annotated_users.csv')

# if i want to reset the indices to get them in order i can do:
df_annotated_users.reset_index(drop=False, inplace=True)
# if drop=False it saves in a column the old indices; inplace=True modifies the dataframe instead of creating a new one

# let's say I want to select (locate) all the tweets that contain in the text the string "shrek", i can use:
df.loc[df["text"].str.contains("shrek")]

# if i want to specify some more complicated conditions on strings i can use regular expressions:
import re
# for example, this is to search if the "user.screen_name" starts with "gian" 
# (flags=re.I means ignore casing; without the ^ then it must contain "gian" but not necessarily at the start):
df.loc[df["user.screen_name"].str.contains("^gian[a-z]*", flags=re.I, regex=True)]



# i can change an attribute for all the rows that satisfy a condition.
# for example i can change all the "AntiVax" in the user annotation to "NoVax":
df.loc[df["user_annotation"]=='Antivax', 'user_annotation'] = 'NoVax'
# the second "user_annotation" is the actual attribute to change. n.b. they
# don't have to be the same, i could for example select all the antivaxxers and change
# their name to douchebage in this way:
df.loc[df["user_annotation"]=='Antivax', 'user.screen_name'] = 'Douchebag'


# let's say i want to know some statistic on our dataset - in our case i don't think
# this is helpful because our value don't represent anything in particular (i.e. are all categorical/nominal values)
# that means for example the mean of all tweet ids is a meaningless number, but let's say we want to calculate it
# for different groups of users, for example i want the mean id for novaxxers, the mean id for provaxxers etcc and compare them.:
# i can do this:
df.groupby(["user_annotation"]).mean()
#this returns all the means for subgroups, grouped by the user_annotation
#and i can also show them sorted for legibility
df.groupby(["user_annotation"]).mean().sort_values("user.id", ascending=False)
#in this way i sort them by their mean value of "user.id"
# other useful groupby functions are groupby.sum() and groupby.count() and as you
# can guess the first one returns the sum of all values and count() returns the
 # number of elements in each group (for each attribute, so if there are nans
 # in an attribute the count will be different)

#this is to count how many elements there are for each user_annotation
df.groupby(["user_annotation"]).count()

# or if i want a cleaner count i could use this trick
df["count"]=1   # i create a count attribute that's 1 for everyone
df.groupby(["user_annotation"]).count()["count"]    # this only shows the counts for the counting colum'n



'''
 GOODBYE!!! ♥♥♥
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$**$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$"   ^$$$$$$F    *$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$     z$$$$$$L    ^$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$    e$$$$$$$$$e  J$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$eee$$$$$$$$$$$$$e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$b$$$$$$$$$$$$$$$$$$*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$)$$$$P"e^$$$F$r*$$$$F"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$d$$$$  "z$$$$"  $$$$%  $3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$*"""*$$$  .$$$$$$ z$$$*   ^$e*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$"     *$$ee$$$$$$$$$$*"     $$$C$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$.      "***$$"*"$$""        $$$$e*$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$b          "$b.$$"          $$$$$b"$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$c.         """            $$$$$$$^$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$e..                     $$$$$$$$^$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$eeee..            J$$$$$$$$b"$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$r          z$$$$$$$$$$r$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"         z$$$$$**$$$$$^$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$*"          z$$$P"   ^*$$$ $$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$*"           .d$$$$       $$$ $$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$"           .e$$$$$F       3$$ $$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$.         .d$$$$$$$         $PJ$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$eeeeeeed$*""""**""         $\$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$                  $d$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$.                 $$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$e.              d$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$eeeeeee$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Gilo94'$$$$


'''


