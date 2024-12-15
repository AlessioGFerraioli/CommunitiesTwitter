# CommunitiesTwitter
![ProVax/NoVax Communities on Twitter](https://github.com/AlessioGFerraioli/CommunitiesTwitter/blob/main/header.jpg?raw=true)

### Network and Communities analysis of Twitter users discussing Covid-19 vaccines in Italy.

This project was done as a collaboration between Alessio Giuseppe Ferraioli and Giancarlo Franco Cuticchia during the course of "Complex Networks" by prof. D. Remondini at the University of Bologna. 

[__<span style="color:red"> Please read: </span>__ A detailed report of the analysis can be read here.](https://drive.google.com/file/d/1jgkCqYHEIYhxFxMG5rc_QwydLsOFgtUN/view?usp=sharing)

### Structure of the Repo

- ```scripts\complete_analysis.py```: The main script containing the complete analysis
- ```scripts\communities.py```: A simpler script that performs the community detection

The folder ```preliminary_scripts``` contains several intermediate scripts and working files that we used to build the final analysis (```scripts\complete_analysis.py```)

The folder ```plots``` contains several visualizations of the data, including:
- ```\centrality_plots```: visualizations of centrality measures on the network, including __weighted and unweighted in-degree and out-degree__, __closeness centrality__, __betwenness centrality__, and a __PCA__ dimensionality reduction of the centrality measures;
- ```\pie_charts```: visualizations of pie charts representing the percentage of __AntiVax, NoVax and ProVax users in each community__;
- ```\network_visualization```: several visualizations of the __network__ and the __communities__ as __graphs__.
- ```\tweets_lenght```: visualizaions of the __number of characters__ in the tweets by users of the three groups
- ```\tweets_per_day```: visualizaions of the __number of tweets per day__ by the users of the three groups.


### The analysis

We build a network as a weighted directed graph in which __users are nodes and edges are tweets__. There is an __edge directed from user A to user B if user A has retweeted a tweet by user B__. The weight of the edge is the total number of times user A has retweeted a tweet by user B.
To filter out some random noise, we remove all the edges with weight less than 3.

We perform network analysis on the graph, highlighting the difference between the three user classes (_AntiVax_, _ProVax_, _Neutral_), such as centrality measures (i.e.: __weighted__ and __unweighted in-degree__, __out-degree__ and __undirected degree__, __closeness__, __betwenness__), __frequency__ of tweets, __length__ of tweets..

We perform a __Leiden__ and a __Louvain community detection__ on the graph, finding the communities of users in the network and check how the three classes are distributed within these communities.

[For more information and the specific on the techniques, theory, and results, a detailed report can be read here.](https://drive.google.com/file/d/1jgkCqYHEIYhxFxMG5rc_QwydLsOFgtUN/view?usp=sharing)

### Info on data

Data is a table in which each row is a single tweet (or retweet), collected in 2021, capturing discussion between italian users regarding the Covid-19 vaccines.

The columns are:

- ```created_at```: date of tweet creation
- ```id```: unique tweet identifier 
- ```text```: text content of the tweet
- ```user.id```: unique user identifier
- ```user.screen_name```: displayed name of the user (might change over time, so it is not recommended for the analysis)
- ```place```: information about the location of the user (often missing)
- ```url```: any URL contained within the tweet
- ```retweeted_status.id```: if the tweet is a retweet, this is the id of the original tweet. Otherwise, this is empty.
- ```retweeted_user.id```: if the tweet is a retweet, this is the id of the user who created the original tweet. Otherwise, this is empty.
- ``` retweeted_url```: if the tweet is a retweet, this is the url contained in the original tweet (if any). Otherwise, this is empty.
- ```annotation```: label indicating the polarity class of the tweet (AntiVax, Neutral, ProVax), produced by manual human annotation. In total, 6508 tweets are labeled.
- ```user_annotation```: label indicating the polarity class of the user (available for users with annotated tweets).
