# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:19:51 2021
 <xcbswbdmnjdg
@author: Utente
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

#%%%%%%%%%%%%   selecting the subset of the dataframe we need


df_user_annotation = df[df['user_annotation'].notna()]

df_user_annotation_retweet = df_user_annotation[df_user_annotation['retweeted_status.user.id'].notna()]


#%%%  creating the graph object


nx.from_pandas_edgelist(df, source='user.id',
                        target='retweeted_status.user.id')